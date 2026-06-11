from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from video_commerce.services.interaction_ingest import api as interaction_ingest_api
from video_commerce.common.models import InteractionType, UserInteractionRequest


class DummyKafkaManager:
    def __init__(self, should_succeed: bool):
        self.should_succeed = should_succeed
        self.calls = []

    async def send_user_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return self.should_succeed


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
