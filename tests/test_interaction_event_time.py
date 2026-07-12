from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from video_commerce.common.event_time import validate_public_event_time
from video_commerce.common.models import InteractionType, UserInteractionRequest
from video_commerce.services.interaction_ingest import api as interaction_ingest_api


class DummyKafkaManager:
    def __init__(self):
        self.calls = []

    async def send_user_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return True


def test_interaction_request_promotes_legacy_timestamp_to_event_time():
    request = UserInteractionRequest(
        user_id="user-1",
        product_id="product-1",
        action=InteractionType.CLICK,
        timestamp=1_700_000_000.0,
    )

    assert request.event_time == 1_700_000_000.0
    assert request.timestamp == 1_700_000_000.0


def test_interaction_request_rejects_conflicting_event_time_aliases():
    with pytest.raises(ValidationError, match="must match"):
        UserInteractionRequest(
            user_id="user-1",
            product_id="product-1",
            action=InteractionType.CLICK,
            event_time=1_700_000_000.0,
            timestamp=1_700_000_001.0,
        )


def test_public_event_time_has_historical_and_future_bounds():
    received_at = 1_700_000_000.0

    assert validate_public_event_time(received_at - 30 * 86400, received_at) == (
        received_at - 30 * 86400
    )
    assert validate_public_event_time(received_at + 5 * 60, received_at) == (
        received_at + 5 * 60
    )
    with pytest.raises(ValueError, match="older than"):
        validate_public_event_time(received_at - 30 * 86400 - 0.001, received_at)
    with pytest.raises(ValueError, match="future"):
        validate_public_event_time(received_at + 5 * 60 + 0.001, received_at)


@pytest.mark.asyncio
async def test_interaction_ingest_stamps_server_receive_time(monkeypatch):
    manager = DummyKafkaManager()
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api.time, "time", lambda: 1_700_000_100.0)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )

    response = await interaction_ingest_api.ingest_interaction(
        SimpleNamespace(state=SimpleNamespace(request_id="request-1")),
        UserInteractionRequest(
            user_id="user-1",
            product_id="product-1",
            action=InteractionType.CLICK,
            event_time=1_700_000_000.0,
        ),
    )

    assert response.status_code == 202
    assert manager.calls == [
        {
            "user_id": "user-1",
            "product_id": "product-1",
            "action": "click",
            "context": {},
            "event_time": 1_700_000_000.0,
            "server_received_at": 1_700_000_100.0,
            "request_id": "request-1",
        }
    ]


@pytest.mark.asyncio
async def test_interaction_ingest_rejects_untrusted_historical_event_time(monkeypatch):
    manager = DummyKafkaManager()
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api.time, "time", lambda: 1_700_000_100.0)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )

    with pytest.raises(interaction_ingest_api.HTTPException) as exc:
        await interaction_ingest_api.ingest_interaction(
            SimpleNamespace(state=SimpleNamespace(request_id="request-2")),
            UserInteractionRequest(
                user_id="user-1",
                product_id="product-1",
                action=InteractionType.CLICK,
                event_time=1_700_000_100.0 - 30 * 86400 - 1.0,
            ),
        )

    assert exc.value.status_code == 422
    assert manager.calls == []
