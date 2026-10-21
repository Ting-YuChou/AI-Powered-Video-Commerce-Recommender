"""Interaction ingest answers 503 fast when Kafka is slow or down."""

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

from video_commerce.common.config import Config, DatabaseConfig, KafkaConfig
from video_commerce.common.models import (
    InteractionType,
    UserInteractionRequest,
    ViewedImpressionRequest,
)
from video_commerce.data_plane import kafka_client as kafka_client_module
from video_commerce.data_plane.kafka_client import (
    KafkaProducerClient,
    KafkaProducerSupervisor,
)
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.services.interaction_ingest import api as ingest_api
from video_commerce.services.interaction_ingest.publish_breaker import PublishBreaker

BUDGET_SECONDS = 0.05


class GatedKafkaManager:
    """Kafka whose sends wait until the test settles each one."""

    def __init__(self):
        self.producer = SimpleNamespace(is_connected=True)
        self.calls = []
        self.outcomes = []
        self.instant = None
        self.raises = None

    async def send_user_interaction(self, **kwargs):
        return await self._send(kwargs)

    async def send_recommendation_view(self, **kwargs):
        return await self._send(kwargs)

    async def _send(self, kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        if self.instant is not None:
            return self.instant
        outcome = asyncio.get_running_loop().create_future()
        self.outcomes.append(outcome)
        return await outcome

    def settle(self, index, published):
        self.outcomes[index].set_result(published)


class Ledger:
    """In-memory copy of SystemStore's interaction claim and lease rules."""

    def __init__(self):
        self.rows = {}
        self.renewals = 0
        self.renew_error = None

    async def claim_interaction_event(
        self, *, event_id, payload_hash, event_payload, lease_seconds
    ):
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

    def _owned(self, event_id, claim_token):
        row = self.rows.get(event_id)
        return row is not None and row["status"] == "pending" and (
            row["lease_owner"] == claim_token
        )

    async def mark_interaction_event_published(self, event_id, *, claim_token):
        if not self._owned(event_id, claim_token):
            return False
        self.rows[event_id].update(
            status="published", lease_owner=None, lease_expires_at=None
        )
        return True

    async def release_interaction_event_claim(self, event_id, *, claim_token):
        if not self._owned(event_id, claim_token):
            return False
        self.rows[event_id].update(lease_owner=None, lease_expires_at=time.monotonic())
        return True

    async def renew_interaction_event_claims(self, claims, *, lease_seconds):
        if self.renew_error is not None:
            raise self.renew_error
        renewed = 0
        for event_id, claim_token in claims:
            if self._owned(event_id, claim_token):
                self.rows[event_id]["lease_expires_at"] = time.monotonic() + lease_seconds
                renewed += 1
        self.renewals += renewed
        return renewed

    async def close(self):
        return None


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.renewals = 0

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)

    async def eval(self, script, numkeys, key, expected, ttl):
        assert script == ingest_api._RENEW_PENDING_REDIS_CLAIM
        if self.values.get(key) != expected:
            return 0
        self.renewals += 1
        return 1


class RecordingSupervisor:
    def __init__(self):
        self.reconnects = 0

    def reconnect_in_background(self):
        self.reconnects += 1


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


async def _drain():
    while ingest_api._late_publishes:
        await asyncio.wait(set(ingest_api._late_publishes), timeout=1)


@pytest_asyncio.fixture
async def ingest(monkeypatch):
    manager = GatedKafkaManager()
    ledger = Ledger()
    supervisor = RecordingSupervisor()
    clock = Clock()
    breaker = PublishBreaker(failure_threshold=5, cooldown_seconds=5.0, clock=clock)
    invalidated = []

    async def record_invalidation(user_id):
        invalidated.append(user_id)

    monkeypatch.setattr(ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(ingest_api, "kafka_supervisor", supervisor)
    monkeypatch.setattr(ingest_api, "system_store", ledger)
    monkeypatch.setattr(ingest_api, "feature_store", None)
    monkeypatch.setattr(ingest_api, "publish_timeout_seconds", BUDGET_SECONDS)
    monkeypatch.setattr(ingest_api, "max_outstanding_publishes", 1000)
    monkeypatch.setattr(ingest_api, "publish_breaker", breaker)
    monkeypatch.setattr(ingest_api, "_outstanding_publishes", set())
    monkeypatch.setattr(ingest_api, "_late_publishes", set())
    monkeypatch.setattr(ingest_api, "_late_claims", {})
    monkeypatch.setattr(ingest_api, "_claim_renewer", None)
    monkeypatch.setattr(ingest_api, "CLAIM_RENEW_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ingest_api, "_invalidate_user_serving_cache", record_invalidation)
    monkeypatch.setattr(
        ingest_api.app.state.runtime,
        "config",
        SimpleNamespace(kafka_config=SimpleNamespace(enable=True)),
    )
    yield SimpleNamespace(
        manager=manager,
        ledger=ledger,
        supervisor=supervisor,
        clock=clock,
        breaker=breaker,
        invalidated=invalidated,
    )
    # Settle sends a test left gated so no task outlives its event loop.
    for outcome in manager.outcomes:
        if not outcome.done():
            outcome.set_result(False)
    await _drain()
    if ingest_api._claim_renewer is not None:
        ingest_api._claim_renewer.cancel()


def _interaction(event_id=None):
    return UserInteractionRequest(
        event_id=event_id or uuid4(),
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
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["retryable"] is True
    assert exc.value.headers["Retry-After"] == "1"
    return exc.value, time.monotonic() - started


def _code(error):
    return error.detail["error"]["code"]


def _sample(name, labels):
    registry = ingest_api.app.state.runtime.observability.registry
    return registry.get_sample_value(name, labels) or 0.0


def _ingest_count(status):
    return _sample(
        "video_commerce_interactions_ingested_total",
        {"action": "click", "status": status},
    )


def _view_count(status):
    return _sample(
        "video_commerce_recommendation_impression_events_total",
        {"stage": "viewed", "status": status},
    )


# ------------------------------------------------------------ timeouts


@pytest.mark.asyncio
async def test_slow_kafka_answers_retryable_503_within_budget(ingest):
    event_id = str(uuid4())
    timeouts_before = _ingest_count("publish_timeout")
    pending_before = _ingest_count("publish_pending")

    error, elapsed = await _expect_503(_interaction(event_id))

    assert elapsed < BUDGET_SECONDS + 0.5
    assert _code(error) == "KAFKA_PUBLISH_TIMEOUT"
    assert error.detail["error"]["request_id"] == "req-budget"
    assert error.detail["event_id"] == event_id
    assert _ingest_count("publish_timeout") == timeouts_before + 1
    # The send may still land, so its claim stays pending instead of released.
    row = ingest.ledger.rows[event_id]
    assert row["status"] == "pending" and row["lease_owner"] is not None

    retry, _ = await _expect_503(_interaction(event_id))
    assert _code(retry) == "INTERACTION_PUBLISH_PENDING"
    assert retry.detail["event_id"] == event_id
    assert _ingest_count("publish_pending") == pending_before + 1
    assert len(ingest.manager.calls) == 1


@pytest.mark.asyncio
async def test_server_generated_event_id_is_returned_for_retries(ingest):
    error, _ = await _expect_503(
        UserInteractionRequest(user_id="u1", product_id="p1", action=InteractionType.CLICK)
    )

    assert error.detail["event_id"] == ingest.manager.calls[0]["event_id"]


@pytest.mark.asyncio
async def test_retry_after_unpublished_timeout_is_accepted_not_duplicate(ingest):
    event_id = str(uuid4())
    failed_before = _ingest_count("late_publish_failed")
    await _expect_503(_interaction(event_id))

    # Kafka gives up on the first send; the claim is released for a retry.
    ingest.manager.settle(0, False)
    await _drain()
    row = ingest.ledger.rows[event_id]
    assert row["status"] == "pending" and row["lease_owner"] is None
    assert _ingest_count("late_publish_failed") == failed_before + 1

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

    ingest.manager.settle(0, True)
    await _drain()

    assert ingest.ledger.rows[event_id]["status"] == "published"
    assert _ingest_count("published_late") == late_before + 1
    assert ingest.invalidated == ["u1"]
    response = await _post(_interaction(event_id))
    assert response.status_code == 202
    assert json.loads(response.body)["duplicate"] is True
    assert len(ingest.manager.calls) == 1


@pytest.mark.asyncio
async def test_late_send_keeps_its_claim_renewed_past_the_lease(ingest, monkeypatch):
    monkeypatch.setattr(ingest_api, "IDEMPOTENCY_PENDING_TTL_SECONDS", 0.05)
    event_id = str(uuid4())
    await _expect_503(_interaction(event_id))

    await asyncio.sleep(0.2)
    assert ingest.ledger.renewals > 0
    retry, _ = await _expect_503(_interaction(event_id))
    assert _code(retry) == "INTERACTION_PUBLISH_PENDING"

    ingest.manager.settle(0, True)
    await _drain()
    renewals = ingest.ledger.renewals
    await asyncio.sleep(0.05)
    assert ingest.ledger.renewals == renewals
    assert ingest_api._late_claims == {}
    assert ingest.ledger.rows[event_id]["status"] == "published"


@pytest.mark.asyncio
async def test_late_landing_after_lapsed_lease_is_counted(ingest, monkeypatch):
    monkeypatch.setattr(ingest_api, "IDEMPOTENCY_PENDING_TTL_SECONDS", 0.05)
    ingest.ledger.renew_error = RuntimeError("ledger down")
    lost_before = _ingest_count("late_publish_lost_lease")
    event_id = str(uuid4())
    await _expect_503(_interaction(event_id))
    await asyncio.sleep(0.1)

    # The lease lapsed, so a retry reclaims and publishes the event_id again.
    ingest.manager.instant = True
    response = await _post(_interaction(event_id))
    assert json.loads(response.body)["duplicate"] is False

    ingest.manager.settle(0, True)
    await _drain()

    assert len(ingest.manager.calls) == 2
    assert _ingest_count("late_publish_lost_lease") == lost_before + 1
    assert ingest.ledger.rows[event_id]["status"] == "published"


# ------------------------------------------------------------ failures


@pytest.mark.asyncio
async def test_publish_that_raises_releases_claim_at_once(ingest):
    event_id = str(uuid4())
    ingest.manager.raises = RuntimeError("serializer broke")
    failed_before = _ingest_count("publish_failed")

    error, _ = await _expect_503(_interaction(event_id))

    assert _code(error) == "KAFKA_PUBLISH_FAILED"
    assert ingest.ledger.rows[event_id]["lease_owner"] is None
    assert _ingest_count("publish_failed") == failed_before + 1
    ingest.manager.raises = None
    ingest.manager.instant = True
    response = await _post(_interaction(event_id))
    assert json.loads(response.body)["duplicate"] is False


@pytest.mark.asyncio
async def test_failed_publish_inside_budget_releases_claim(ingest):
    event_id = str(uuid4())
    ingest.manager.instant = False

    error, _ = await _expect_503(_interaction(event_id))

    assert _code(error) == "KAFKA_PUBLISH_FAILED"
    assert ingest.ledger.rows[event_id]["lease_owner"] is None


@pytest.mark.asyncio
async def test_client_disconnect_leaves_send_to_settle_the_claim(ingest):
    event_id = str(uuid4())
    ingest_api.publish_timeout_seconds = 5.0
    request = asyncio.ensure_future(_post(_interaction(event_id)))
    await asyncio.sleep(0.01)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert ingest.ledger.rows[event_id]["lease_owner"] is not None

    ingest.manager.settle(0, False)
    await _drain()
    assert ingest.ledger.rows[event_id]["lease_owner"] is None
    assert ingest.breaker._failures == 1


@pytest.mark.asyncio
async def test_disconnected_producer_answers_503_without_claim_or_send(ingest):
    ingest.manager.producer.is_connected = False
    event_id = str(uuid4())

    error, elapsed = await _expect_503(_interaction(event_id))

    assert elapsed < BUDGET_SECONDS
    assert _code(error) == "KAFKA_UNAVAILABLE"
    assert error.detail["event_id"] == event_id
    assert ingest.manager.calls == []
    assert event_id not in ingest.ledger.rows
    assert ingest.supervisor.reconnects == 1

    ingest.manager.producer.is_connected = True
    ingest.manager.instant = True
    response = await _post(_interaction(event_id))
    assert response.status_code == 202
    assert json.loads(response.body)["duplicate"] is False


# ------------------------------------------------------------ breaker and cap


@pytest.mark.asyncio
async def test_repeated_timeouts_trip_breaker_then_one_probe_recovers(ingest):
    results = await asyncio.gather(
        *(_post(_interaction()) for _ in range(5)), return_exceptions=True
    )
    assert [_code(r) for r in results] == ["KAFKA_PUBLISH_TIMEOUT"] * 5
    assert ingest.breaker.tripped

    # Open: refused at once, before the ledger or Kafka is touched.
    event_id = str(uuid4())
    error, elapsed = await _expect_503(_interaction(event_id))
    assert _code(error) == "KAFKA_UNAVAILABLE"
    assert elapsed < BUDGET_SECONDS
    assert event_id not in ingest.ledger.rows
    assert len(ingest.manager.calls) == 5

    # After the cooldown one probe goes through while others still fail fast.
    ingest.clock.now += 5.0
    ingest_api.publish_timeout_seconds = 5.0
    probe = asyncio.ensure_future(_post(_interaction()))
    await asyncio.sleep(0.01)
    blocked, _ = await _expect_503(_interaction())
    assert _code(blocked) == "KAFKA_UNAVAILABLE"
    assert len(ingest.manager.calls) == 6

    ingest.manager.settle(5, True)
    assert (await probe).status_code == 202
    assert not ingest.breaker.tripped
    ingest.manager.instant = True
    assert (await _post(_interaction())).status_code == 202


@pytest.mark.asyncio
async def test_failed_probe_restarts_the_cooldown(ingest):
    for _ in range(5):
        ingest.breaker.record_failure()
    ingest.clock.now += 5.0
    ingest.manager.instant = False

    error, _ = await _expect_503(_interaction())

    assert _code(error) == "KAFKA_PUBLISH_FAILED"
    assert ingest.breaker.blocking()


@pytest.mark.asyncio
async def test_unresolved_publish_cap_fails_fast(ingest):
    ingest_api.max_outstanding_publishes = 3
    ingest_api.publish_timeout_seconds = 0.2

    results = await asyncio.gather(
        *(_post(_interaction()) for _ in range(5)), return_exceptions=True
    )

    codes = sorted(_code(r) for r in results)
    assert codes == ["KAFKA_PUBLISH_TIMEOUT"] * 3 + ["KAFKA_UNAVAILABLE"] * 2
    assert len(ingest.manager.calls) == 3

    # Sends that resolve free their slots.
    for index in range(3):
        ingest.manager.settle(index, False)
    await _drain()
    ingest.breaker.record_success()
    ingest.manager.instant = True
    assert (await _post(_interaction())).status_code == 202


def test_breaker_admits_a_single_probe_per_cooldown():
    clock = Clock()
    breaker = PublishBreaker(failure_threshold=2, cooldown_seconds=1.0, clock=clock)
    assert breaker.admit()
    breaker.record_failure()
    assert breaker.admit()
    breaker.record_failure()
    assert breaker.blocking() and not breaker.admit()

    clock.now += 1.0
    assert breaker.admit()
    assert not breaker.admit()
    breaker.release_probe()
    assert breaker.admit()
    breaker.record_success()
    assert not breaker.tripped and breaker.admit()


# ------------------------------------------------------------ shutdown, Redis


@pytest.mark.asyncio
async def test_shutdown_drains_late_sends_before_closing_stores(ingest, monkeypatch):
    monkeypatch.setattr(ingest_api, "kafka_supervisor", None)
    event_id = str(uuid4())
    await _expect_503(_interaction(event_id))
    asyncio.get_running_loop().call_later(0.05, ingest.manager.settle, 0, True)

    await ingest_api.shutdown_event()

    assert ingest.ledger.rows[event_id]["status"] == "published"
    assert ingest_api._late_publishes == set()


@pytest.mark.asyncio
async def test_shutdown_stops_waiting_for_sends_that_never_resolve(ingest, monkeypatch):
    monkeypatch.setattr(ingest_api, "kafka_supervisor", None)
    monkeypatch.setattr(ingest_api, "LATE_PUBLISH_DRAIN_SECONDS", 0.05)
    await _expect_503(_interaction())

    started = time.monotonic()
    await ingest_api.shutdown_event()

    assert time.monotonic() - started < 1
    with pytest.raises(asyncio.CancelledError):
        await ingest_api._claim_renewer


@pytest.mark.asyncio
async def test_redis_ledger_keeps_late_claim_pending_then_releases_it(ingest, monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(ingest_api, "system_store", None)
    monkeypatch.setattr(
        ingest_api,
        "feature_store",
        SimpleNamespace(redis_client=redis, invalidate_user_serving_cache=None),
    )
    event_id = str(uuid4())
    key = f"interaction:idempotency:{event_id}"

    await _expect_503(_interaction(event_id))
    assert json.loads(redis.values[key])["status"] == "pending"
    await asyncio.sleep(0.05)
    assert redis.renewals > 0
    retry, _ = await _expect_503(_interaction(event_id))
    assert _code(retry) == "INTERACTION_PUBLISH_PENDING"

    ingest.manager.settle(0, False)
    await _drain()
    assert key not in redis.values

    ingest.manager.instant = True
    response = await _post(_interaction(event_id))
    assert json.loads(response.body)["duplicate"] is False
    assert json.loads(redis.values[key])["status"] == "published"


# ------------------------------------------------------------ viewed impressions


def _views(count=2):
    return ViewedImpressionRequest(
        impression_id=uuid4(),
        viewed_at=time.time(),
        items=[
            {"event_id": uuid4(), "product_id": f"p{index}", "position": index}
            for index in range(1, count + 1)
        ],
    )


class SlateStore:
    async def get_recommendation_impression_slate(self, impression_id):
        return {"user_id": "u1", "items": {"p1": 1, "p2": 2}}

    async def record_recommendation_impression_views(self, **kwargs):
        return 1


async def _post_views(payload):
    with pytest.raises(HTTPException) as exc:
        await ingest_api.ingest_viewed_impression(_http_request(), payload)
    assert exc.value.status_code == 503
    return exc.value


@pytest.mark.asyncio
async def test_slow_views_share_one_deadline_and_settle_late(ingest):
    ingest_api.system_store = SlateStore()
    payload = _views()
    timeouts_before = _view_count("publish_timeout")
    late_before = _view_count("published_late")

    started = time.monotonic()
    error = await _post_views(payload)

    assert time.monotonic() - started < BUDGET_SECONDS + 0.5
    assert _code(error) == "KAFKA_PUBLISH_TIMEOUT"
    assert error.detail["impression_id"] == str(payload.impression_id)
    # Both items were in flight under the one deadline, not sent one by one.
    assert len(ingest.manager.calls) == 2
    assert _view_count("publish_timeout") == timeouts_before + 1

    ingest.manager.settle(0, True)
    ingest.manager.settle(1, True)
    await _drain()
    assert _view_count("published_late") == late_before + 2


@pytest.mark.asyncio
async def test_failed_view_publish_is_counted(ingest):
    ingest_api.system_store = SlateStore()
    ingest.manager.instant = False
    failed_before = _view_count("publish_failed")

    error = await _post_views(_views())

    assert _code(error) == "KAFKA_PUBLISH_FAILED"
    assert _view_count("publish_failed") == failed_before + 1


@pytest.mark.asyncio
async def test_disconnected_producer_rejects_viewed_impressions(ingest):
    ingest.manager.producer.is_connected = False
    unavailable_before = _view_count("kafka_unavailable")

    error = await _post_views(_views(1))

    assert _code(error) == "KAFKA_UNAVAILABLE"
    assert ingest.manager.calls == []
    assert _view_count("kafka_unavailable") == unavailable_before + 1


# ------------------------------------------------------------ producer and supervisor


class _FakeAioProducer:
    def __init__(self, exc=None):
        self.exc = exc
        self.stopped = False

    async def send_and_wait(self, **kwargs):
        if self.exc is not None:
            raise self.exc

    async def stop(self):
        self.stopped = True


def _supervised_client(exc):
    client = KafkaProducerClient(KafkaConfig(enable=True))
    client.producer = aio = _FakeAioProducer(exc)
    client.is_connected = True
    unreachable = []
    client.on_unreachable = lambda: unreachable.append(True)
    return client, aio, unreachable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        kafka_client_module.KafkaTimeoutError(),
        kafka_client_module.KafkaConnectionError(),
    ],
)
async def test_unreachable_broker_marks_supervised_producer_down(exc):
    client, aio, unreachable = _supervised_client(exc)

    assert await client.send("topic", {"a": 1}) is False
    await asyncio.sleep(0)

    assert client.is_connected is False and client.producer is None
    assert unreachable == [True]
    assert aio.stopped


@pytest.mark.asyncio
async def test_non_connection_failure_keeps_producer(monkeypatch):
    client, aio, unreachable = _supervised_client(
        kafka_client_module.MessageSizeTooLargeError()
    )

    assert await client.send("topic", {"a": 1}) is False

    assert client.is_connected and client.producer is aio
    assert unreachable == []


@pytest.mark.asyncio
async def test_unsupervised_producer_keeps_its_old_behaviour():
    client = KafkaProducerClient(KafkaConfig(enable=True))
    client.producer = aio = _FakeAioProducer(kafka_client_module.KafkaTimeoutError())
    client.is_connected = True

    assert await client.send("topic", {"a": 1}) is False

    assert client.is_connected and client.producer is aio


@pytest.mark.asyncio
async def test_supervisor_registers_reconnect_hook(monkeypatch):
    manager = SimpleNamespace(producer=SimpleNamespace(on_unreachable=None))

    async def fake_init_kafka(config, observability=None):
        return manager

    monkeypatch.setattr(kafka_client_module, "init_kafka", fake_init_kafka)
    supervisor = KafkaProducerSupervisor(KafkaConfig(enable=True), service_name="test")

    assert await supervisor.start() is manager
    assert manager.producer.on_unreachable == supervisor.reconnect_in_background


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


def test_publish_timeout_must_stay_below_gateway_read_timeout(monkeypatch, caplog):
    monkeypatch.setenv("SERVICE_INTERACTION_INGEST_PUBLISH_TIMEOUT_MS", "10000")
    monkeypatch.setenv("SERVICE_PROXY_READ_TIMEOUT_SECONDS", "10")

    with pytest.raises(ValueError, match="Invalid configuration"):
        Config()
    assert "SERVICE_INTERACTION_INGEST_PUBLISH_TIMEOUT_MS must be below" in caplog.text


# ------------------------------------------------------------ Postgres ledger


async def _postgres_store():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres ledger integration test")
    store = SystemStore(
        DatabaseConfig(enable=True, url=database_url, auto_create_schema=True)
    )
    await store.initialize()
    return store


async def _delete_ledger_rows(store, *event_ids):
    async with store.session_factory.begin() as session:
        for event_id in event_ids:
            await session.execute(
                text("DELETE FROM interaction_idempotency_ledger WHERE event_id=:event_id"),
                {"event_id": event_id},
            )


@pytest.mark.asyncio
async def test_postgres_ledger_retry_after_unpublished_timeout(ingest):
    store = await _postgres_store()
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
        assert _code(retry) == "INTERACTION_PUBLISH_PENDING"

        ingest.manager.settle(0, False)
        await _drain()
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
        await _delete_ledger_rows(store, event_id)
        await store.close()


@pytest.mark.asyncio
async def test_postgres_renewal_extends_only_the_owners_pending_claim():
    store = await _postgres_store()
    owned, other = str(uuid4()), str(uuid4())
    payload = {"user_id": "u1", "product_id": "p1", "action": "click"}
    try:
        claims = {}
        for event_id in (owned, other):
            claims[event_id] = await store.claim_interaction_event(
                event_id=event_id,
                payload_hash="a" * 64,
                event_payload=payload,
                lease_seconds=1,
            )

        renewed = await store.renew_interaction_event_claims(
            [(owned, claims[owned]["claim_token"]), (other, "not-the-owner")],
            lease_seconds=60,
        )
        await asyncio.sleep(1.2)

        assert renewed == 1
        for event_id, expected in ((owned, "pending"), (other, "claimed")):
            again = await store.claim_interaction_event(
                event_id=event_id,
                payload_hash="a" * 64,
                event_payload=payload,
                lease_seconds=30,
            )
            assert again["status"] == expected
        assert await store.renew_interaction_event_claims([], lease_seconds=60) == 0
    finally:
        await _delete_ledger_rows(store, owned, other)
        await store.close()
