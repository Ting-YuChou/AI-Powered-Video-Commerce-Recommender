from types import SimpleNamespace
import asyncio
import json
import os
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from video_commerce.services.interaction_ingest import api as interaction_ingest_api
from video_commerce.common.config import DatabaseConfig
from video_commerce.common.models import (
    InteractionType,
    UserInteractionRequest,
    ViewedImpressionRequest,
)
from video_commerce.data_plane.system_store import SystemStore


class DummyKafkaManager:
    def __init__(self, should_succeed: bool):
        self.should_succeed = should_succeed
        self.calls = []

    async def send_user_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return self.should_succeed

    async def send_recommendation_view(self, **kwargs):
        self.calls.append(kwargs)
        return self.should_succeed


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)


class FakeFeatureStore:
    def __init__(self):
        self.redis_client = FakeRedis()

    async def invalidate_user_serving_cache(self, user_id):
        return None


@pytest.mark.asyncio
async def test_interaction_ingest_returns_503_when_kafka_unavailable(monkeypatch):
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", None)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )

    request = SimpleNamespace(state=SimpleNamespace(request_id="req-1"))
    payload = UserInteractionRequest(
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
        context={"page": "home"},
    )

    with pytest.raises(HTTPException) as exc:
        await interaction_ingest_api.ingest_interaction(request, payload)

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_interaction_ingest_propagates_request_id(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )

    request = SimpleNamespace(state=SimpleNamespace(request_id="req-42"))
    payload = UserInteractionRequest(
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
        context={"page": "home"},
    )

    response = await interaction_ingest_api.ingest_interaction(request, payload)

    assert response.status_code == 202
    assert manager.calls[0]["request_id"] == "req-42"


@pytest.mark.asyncio
async def test_interaction_ingest_preserves_impression_attribution_context(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )

    request = SimpleNamespace(state=SimpleNamespace(request_id="req-43"))
    payload = UserInteractionRequest(
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
        context={
            "impression_id": "imp-1",
            "recommendation_position": 3,
            "recommendation_ranking_score": 0.72,
            "recommendation_source": "two_tower",
        },
    )

    response = await interaction_ingest_api.ingest_interaction(request, payload)

    assert response.status_code == 202
    assert manager.calls[0]["context"]["impression_id"] == "imp-1"
    assert manager.calls[0]["context"]["recommendation_position"] == 3
    assert manager.calls[0]["context"]["recommendation_source"] == "two_tower"


@pytest.mark.asyncio
async def test_interaction_event_id_is_idempotent_and_detects_collision(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)
    store = FakeFeatureStore()
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api, "feature_store", store)
    monkeypatch.setattr(interaction_ingest_api, "system_store", None)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )
    request = SimpleNamespace(state=SimpleNamespace(request_id="req-idempotent"))
    event_id = uuid4()
    payload = UserInteractionRequest(
        event_id=event_id,
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
        context={"page": "home"},
    )

    invalidated_users = []

    async def record_invalidation(user_id):
        invalidated_users.append(user_id)

    monkeypatch.setattr(
        interaction_ingest_api,
        "_invalidate_user_serving_cache",
        record_invalidation,
    )

    first = await interaction_ingest_api.ingest_interaction(request, payload)
    second = await interaction_ingest_api.ingest_interaction(request, payload)
    await asyncio.sleep(0)

    assert len(manager.calls) == 1
    assert json.loads(first.body)["duplicate"] is False
    assert json.loads(second.body)["duplicate"] is True
    assert invalidated_users == ["u1"]
    collision = payload.copy(update={"product_id": "p2"})
    with pytest.raises(HTTPException) as exc:
        await interaction_ingest_api.ingest_interaction(request, collision)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_postgres_interaction_claim_is_completed_with_lease_owner(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)

    class Ledger:
        def __init__(self):
            self.marked = []

        async def claim_interaction_event(self, **kwargs):
            return {
                "status": "claimed",
                "claim_token": "owner-token",
                "event_payload": kwargs["event_payload"],
            }

        async def mark_interaction_event_published(self, event_id, *, claim_token):
            self.marked.append((event_id, claim_token))
            return True

    ledger = Ledger()
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api, "system_store", ledger)
    monkeypatch.setattr(interaction_ingest_api, "feature_store", FakeFeatureStore())
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )
    request = SimpleNamespace(state=SimpleNamespace(request_id="req-lease"))
    event_id = uuid4()

    response = await interaction_ingest_api.ingest_interaction(
        request,
        UserInteractionRequest(
            event_id=event_id,
            user_id="u1",
            product_id="p1",
            action=InteractionType.CLICK,
        ),
    )
    await asyncio.sleep(0)

    assert response.status_code == 202
    assert ledger.marked == [(str(event_id), "owner-token")]


@pytest.mark.asyncio
async def test_viewed_impression_binds_user_and_items_to_durable_slate(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)

    class Store:
        async def get_recommendation_impression_slate(self, impression_id):
            return {"user_id": "durable-user", "items": {"p1": 1}}

        async def record_recommendation_impression_views(self, **kwargs):
            return 1

    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api, "system_store", Store())
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )
    request = SimpleNamespace(state=SimpleNamespace(request_id="req-view"))
    payload = ViewedImpressionRequest(
        impression_id=uuid4(),
        viewed_at=interaction_ingest_api.time.time(),
        items=[{"event_id": uuid4(), "product_id": "p1", "position": 1}],
        context={"user_id": "spoofed-user"},
    )

    response = await interaction_ingest_api.ingest_viewed_impression(request, payload)

    assert response.status_code == 202
    assert manager.calls[0]["user_id"] == "durable-user"


@pytest.mark.asyncio
async def test_viewed_impression_isolates_item_outside_durable_slate(monkeypatch):
    manager = DummyKafkaManager(should_succeed=True)

    class Store:
        async def get_recommendation_impression_slate(self, impression_id):
            return {"user_id": "u1", "items": {"p1": 1}}

    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api, "system_store", Store())
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )
    payload = ViewedImpressionRequest(
        impression_id=uuid4(),
        viewed_at=interaction_ingest_api.time.time(),
        items=[{"event_id": uuid4(), "product_id": "p2", "position": 1}],
    )

    response = await interaction_ingest_api.ingest_viewed_impression(
        SimpleNamespace(state=SimpleNamespace(request_id="req-orphan")), payload
    )

    assert json.loads(response.body)["status"] == "isolated"
    assert manager.calls == []


@pytest.mark.asyncio
async def test_postgres_interaction_ledger_rejects_stale_lease_owner():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres ledger integration test")
    store = SystemStore(
        DatabaseConfig(enable=True, url=database_url, auto_create_schema=True)
    )
    await store.initialize()
    event_id = str(uuid4())
    payload = {"user_id": "u1", "product_id": "p1", "action": "click"}
    try:
        first = await store.claim_interaction_event(
            event_id=event_id,
            payload_hash="a" * 64,
            event_payload=payload,
            lease_seconds=30,
        )
        assert first["status"] == "claimed"
        assert (
            await store.claim_interaction_event(
                event_id=event_id,
                payload_hash="a" * 64,
                event_payload=payload,
                lease_seconds=30,
            )
        )["status"] == "pending"
        assert await store.release_interaction_event_claim(
            event_id, claim_token=first["claim_token"]
        )

        second = await store.claim_interaction_event(
            event_id=event_id,
            payload_hash="a" * 64,
            event_payload=payload,
            lease_seconds=30,
        )
        assert second["status"] == "claimed"
        assert not await store.mark_interaction_event_published(
            event_id, claim_token=first["claim_token"]
        )
        assert await store.mark_interaction_event_published(
            event_id, claim_token=second["claim_token"]
        )
        duplicate = await store.claim_interaction_event(
            event_id=event_id,
            payload_hash="a" * 64,
            event_payload=payload,
            lease_seconds=30,
        )
        assert duplicate["status"] == "duplicate"
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text("DELETE FROM interaction_idempotency_ledger WHERE event_id=:event_id"),
                {"event_id": event_id},
            )
        await store.close()
