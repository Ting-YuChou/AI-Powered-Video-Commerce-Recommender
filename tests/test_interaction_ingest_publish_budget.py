"""Interaction ingest answers 503 fast when Kafka is slow or disconnected."""

import asyncio
import json
import os
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import text

from video_commerce.common.config import DatabaseConfig, KafkaConfig
from video_commerce.common.models import (
    InteractionType,
    UserInteractionRequest,
    ViewedImpressionRequest,
)
from video_commerce.data_plane.kafka_client import KafkaProducerSupervisor
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.services.interaction_ingest import api as ingest_api

BUDGET_SECONDS = 0.05


class GatedKafkaManager:
    """Kafka whose sends wait until the test lets each one finish."""

    def __init__(self):
        self.producer = SimpleNamespace(is_connected=True)
        self.calls = []
        self.outcomes = []
        self.instant = None

    async def send_user_interaction(self, **kwargs):
        return await self._send(kwargs)

    async def send_recommendation_view(self, **kwargs):
        return await self._send(kwargs)

    async def _send(self, kwargs):
        self.calls.append(kwargs)
        if self.instant is not None:
            return self.instant
        outcome = asyncio.get_running_loop().create_future()
        self.outcomes.append(outcome)
        return await outcome


class Ledger:
    """In-memory copy of SystemStore's interaction claim/lease rules."""

    def __init__(self):
        self.rows = {}
        self.lease_seconds = []

    async def claim_interaction_event(
        self, *, event_id, payload_hash, event_payload, lease_seconds
    ):
        self.lease_seconds.append(lease_seconds)
        now = time.monotonic()
        row = self.rows.get(event_id)
        inserted = row is None
        if inserted:
            row = self.rows[event_id] = {
                "payload_hash": payload_hash,
                "event_payload": dict(event_payload),
                "status": "pending",
            }
        if row["payload_hash"] != payload_hash:
            return {"status": "collision"}
        if row["status"] == "published":
            return {"status": "duplicate", "event_payload": row["event_payload"]}
        if not inserted and (row.get("lease_expires_at") or 0) > now:
            return {"status": "pending"}
        row["lease_owner"] = token = uuid4().hex
        row["lease_expires_at"] = now + lease_seconds
        return {
            "status": "claimed",
            "claim_token": token,
            "event_payload": row["event_payload"],
        }

    async def mark_interaction_event_published(self, event_id, *, claim_token):
        row = self.rows.get(event_id)
        if not row or row["status"] != "pending" or row["lease_owner"] != claim_token:
            return False
        row.update(status="published", lease_owner=None, lease_expires_at=None)
        return True

    async def release_interaction_event_claim(self, event_id, *, claim_token):
        row = self.rows.get(event_id)
        if not row or row["status"] != "pending" or row["lease_owner"] != claim_token:
            return False
        row.update(lease_owner=None, lease_expires_at=time.monotonic())
        return True


class RecordingSupervisor:
    def __init__(self):
        self.reconnects = 0

    def reconnect_in_background(self):
        self.reconnects += 1


@pytest_asyncio.fixture
async def ingest(monkeypatch):
    manager = GatedKafkaManager()
    ledger = Ledger()
    supervisor = RecordingSupervisor()
    invalidated = []

    async def record_invalidation(user_id):
        invalidated.append(user_id)

    monkeypatch.setattr(ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(ingest_api, "kafka_supervisor", supervisor)
    monkeypatch.setattr(ingest_api, "system_store", ledger)
    monkeypatch.setattr(ingest_api, "feature_store", None)
    monkeypatch.setattr(ingest_api, "publish_timeout_seconds", BUDGET_SECONDS)
    monkeypatch.setattr(ingest_api, "_late_publishes", set())
    monkeypatch.setattr(ingest_api, "_invalidate_user_serving_cache", record_invalidation)
    monkeypatch.setattr(
        ingest_api.app.state.runtime,
        "config",
        SimpleNamespace(kafka_config=SimpleNamespace(enable=True)),
    )
    yield SimpleNamespace(
        manager=manager, ledger=ledger, supervisor=supervisor, invalidated=invalidated
    )
    # Settle sends a test left gated so no task outlives its event loop.
    for outcome in manager.outcomes:
        if not outcome.done():
            outcome.set_result(False)
    await _drain_late_publishes()


def _interaction(event_id):
    return UserInteractionRequest(
        event_id=event_id,
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
        context={"page": "home"},
    )


def _http_request():
    return SimpleNamespace(state=SimpleNamespace(request_id="req-budget"))


async def _post(payload):
    return await ingest_api.ingest_interaction(_http_request(), payload)


async def _expect_503(payload):
    started = time.monotonic()
    with pytest.raises(HTTPException) as exc:
        await _post(payload)
    return exc.value, time.monotonic() - started


async def _drain_late_publishes():
    if ingest_api._late_publishes:
        await asyncio.wait(set(ingest_api._late_publishes), timeout=1)


def _ingest_count(status):
    registry = ingest_api.app.state.runtime.observability.registry
    return (
        registry.get_sample_value(
            "video_commerce_interactions_ingested_total",
            {"action": "click", "status": status},
        )
        or 0.0
    )


@pytest.mark.asyncio
async def test_slow_kafka_answers_retryable_503_within_budget(ingest):
    event_id = str(uuid4())
    timeouts_before = _ingest_count("publish_timeout")

    error, elapsed = await _expect_503(_interaction(event_id))

    assert error.status_code == 503
    assert elapsed < BUDGET_SECONDS + 0.5
    assert error.detail["error"]["code"] == "KAFKA_PUBLISH_TIMEOUT"
    assert error.detail["error"]["retryable"] is True
    assert error.detail["error"]["request_id"] == "req-budget"
    assert error.headers["Retry-After"] == "1"
    assert _ingest_count("publish_timeout") == timeouts_before + 1
    # The send may still land, so its claim stays pending instead of released.
    row = ingest.ledger.rows[event_id]
    assert row["status"] == "pending" and row["lease_owner"] is not None

    retry, _ = await _expect_503(_interaction(event_id))
    assert "still publishing" in retry.detail
    assert len(ingest.manager.calls) == 1


@pytest.mark.asyncio
async def test_retry_after_unpublished_timeout_is_accepted_not_duplicate(ingest):
    event_id = str(uuid4())
    await _expect_503(_interaction(event_id))

    # Kafka gives up on the first send; the claim is released for a retry.
    ingest.manager.outcomes[0].set_result(False)
    await _drain_late_publishes()
    row = ingest.ledger.rows[event_id]
    assert row["status"] == "pending" and row["lease_owner"] is None

    ingest.manager.instant = True
    response = await _post(_interaction(event_id))

    assert response.status_code == 202
    assert json.loads(response.body)["duplicate"] is False
    assert len(ingest.manager.calls) == 2
    assert ingest.ledger.rows[event_id]["status"] == "published"


@pytest.mark.asyncio
async def test_send_that_lands_after_timeout_marks_ledger_published(ingest):
    event_id = str(uuid4())
    late_before = _ingest_count("published_late")
    await _expect_503(_interaction(event_id))

    ingest.manager.outcomes[0].set_result(True)
    await _drain_late_publishes()

    assert ingest.ledger.rows[event_id]["status"] == "published"
    assert _ingest_count("published_late") == late_before + 1
    assert ingest.invalidated == ["u1"]
    response = await _post(_interaction(event_id))
    assert response.status_code == 202
    assert json.loads(response.body)["duplicate"] is True
    assert len(ingest.manager.calls) == 1


@pytest.mark.asyncio
async def test_disconnected_producer_answers_503_without_claim_or_send(ingest):
    ingest.manager.producer.is_connected = False
    event_id = str(uuid4())

    error, elapsed = await _expect_503(_interaction(event_id))

    assert elapsed < BUDGET_SECONDS
    assert error.detail["error"]["code"] == "KAFKA_UNAVAILABLE"
    assert error.detail["error"]["retryable"] is True
    assert ingest.manager.calls == []
    assert event_id not in ingest.ledger.rows
    assert ingest.supervisor.reconnects == 1

    ingest.manager.producer.is_connected = True
    ingest.manager.instant = True
    response = await _post(_interaction(event_id))
    assert response.status_code == 202
    assert json.loads(response.body)["duplicate"] is False


@pytest.mark.asyncio
async def test_failed_publish_inside_budget_releases_claim(ingest):
    event_id = str(uuid4())
    ingest.manager.instant = False

    error, _ = await _expect_503(_interaction(event_id))

    assert error.detail["error"]["code"] == "KAFKA_PUBLISH_FAILED"
    assert ingest.ledger.rows[event_id]["lease_owner"] is None


@pytest.mark.asyncio
async def test_pending_lease_outlives_a_late_kafka_send(ingest):
    assert ingest_api._pending_lease_seconds(30000) == 65
    assert ingest_api._pending_lease_seconds(1000) == (
        ingest_api.IDEMPOTENCY_PENDING_TTL_SECONDS
    )
    ingest.manager.instant = True
    await _post(_interaction(str(uuid4())))
    assert ingest.ledger.lease_seconds == [ingest_api.pending_lease_seconds]


@pytest.mark.asyncio
async def test_slow_viewed_impression_publish_answers_503_within_budget(ingest):
    class SlateStore:
        async def get_recommendation_impression_slate(self, impression_id):
            return {"user_id": "u1", "items": {"p1": 1, "p2": 2}}

    ingest_api.system_store = SlateStore()
    payload = ViewedImpressionRequest(
        impression_id=uuid4(),
        viewed_at=time.time(),
        items=[
            {"event_id": uuid4(), "product_id": "p1", "position": 1},
            {"event_id": uuid4(), "product_id": "p2", "position": 2},
        ],
    )

    started = time.monotonic()
    with pytest.raises(HTTPException) as exc:
        await ingest_api.ingest_viewed_impression(_http_request(), payload)

    assert time.monotonic() - started < BUDGET_SECONDS + 0.5
    assert exc.value.detail["error"]["code"] == "KAFKA_PUBLISH_TIMEOUT"
    assert len(ingest.manager.calls) == 1


@pytest.mark.asyncio
async def test_disconnected_producer_rejects_viewed_impressions(ingest):
    ingest.manager.producer.is_connected = False
    payload = ViewedImpressionRequest(
        impression_id=uuid4(),
        viewed_at=time.time(),
        items=[{"event_id": uuid4(), "product_id": "p1", "position": 1}],
    )

    with pytest.raises(HTTPException) as exc:
        await ingest_api.ingest_viewed_impression(_http_request(), payload)

    assert exc.value.detail["error"]["code"] == "KAFKA_UNAVAILABLE"
    assert ingest.manager.calls == []


@pytest.mark.asyncio
async def test_supervisor_reconnects_dropped_producer_once():
    class FlakyProducer:
        def __init__(self):
            self.is_connected = False
            self.starts = 0

        async def start(self):
            self.starts += 1
            if self.starts == 1:
                raise ConnectionError("broker down")
            self.is_connected = True

    supervisor = KafkaProducerSupervisor(
        KafkaConfig(enable=True),
        service_name="test",
        initial_backoff_seconds=0.01,
    )
    supervisor.manager = SimpleNamespace(producer=FlakyProducer())

    supervisor.reconnect_in_background()
    first_task = supervisor._task
    supervisor.reconnect_in_background()
    assert supervisor._task is first_task
    await asyncio.wait_for(first_task, timeout=1)

    assert supervisor.manager.producer.is_connected
    assert supervisor.manager.producer.starts == 2
    supervisor.reconnect_in_background()
    assert supervisor._task is first_task


@pytest.mark.asyncio
async def test_postgres_ledger_retry_after_unpublished_timeout(ingest):
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres ledger integration test")
    store = SystemStore(
        DatabaseConfig(enable=True, url=database_url, auto_create_schema=True)
    )
    await store.initialize()
    ingest_api.system_store = store
    event_id = str(uuid4())

    async def ledger_row():
        async with store.session_factory.begin() as session:
            return (
                await session.execute(
                    text(
                        "SELECT status, lease_owner FROM interaction_idempotency_ledger "
                        "WHERE event_id=:event_id"
                    ),
                    {"event_id": event_id},
                )
            ).one()

    try:
        await _expect_503(_interaction(event_id))
        status, owner = await ledger_row()
        assert status == "pending" and owner is not None
        retry, _ = await _expect_503(_interaction(event_id))
        assert "still publishing" in retry.detail

        ingest.manager.outcomes[0].set_result(False)
        await _drain_late_publishes()
        status, owner = await ledger_row()
        assert status == "pending" and owner is None

        ingest.manager.instant = True
        response = await _post(_interaction(event_id))
        assert response.status_code == 202
        assert json.loads(response.body)["duplicate"] is False
        assert (await ledger_row())[0] == "published"
        duplicate = await _post(_interaction(event_id))
        assert json.loads(duplicate.body)["duplicate"] is True
        assert len(ingest.manager.calls) == 2
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text("DELETE FROM interaction_idempotency_ledger WHERE event_id=:event_id"),
                {"event_id": event_id},
            )
        await store.close()
