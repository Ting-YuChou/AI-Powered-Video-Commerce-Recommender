from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from video_commerce.common.event_time import (
    clamp_public_event_time,
    validate_public_event_time,
)
from video_commerce.common.models import InteractionType, UserInteractionRequest
from video_commerce.services.interaction_ingest import api as interaction_ingest_api


class DummyKafkaManager:
    def __init__(self):
        self.calls = []
        self.producer = SimpleNamespace(is_connected=True)

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
    assert len(manager.calls) == 1
    assert UUID(manager.calls[0].pop("event_id"))
    assert manager.calls == [{
            "user_id": "user-1",
            "product_id": "product-1",
            "action": "click",
            "context": {},
            "event_time": 1_700_000_000.0,
            "server_received_at": 1_700_000_100.0,
            "request_id": "request-1",
            "client_event_time": 1_700_000_000.0,
        }]


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


def test_clamp_public_event_time_bounds_client_time_to_server_window():
    assert clamp_public_event_time(160.0, upper_bound=100.0) == 100.0
    assert clamp_public_event_time(40.0, upper_bound=100.0) == 40.0
    assert clamp_public_event_time(40.0, upper_bound=100.0, lower_bound=50.0) == 50.0
    assert clamp_public_event_time(70.0, upper_bound=100.0, lower_bound=50.0) == 70.0
    # Disagreeing server clocks: the view stays ordered after its serve.
    assert clamp_public_event_time(70.0, upper_bound=100.0, lower_bound=120.0) == 120.0


def _ingest_with_clock(monkeypatch, now):
    manager = DummyKafkaManager()
    monkeypatch.setattr(interaction_ingest_api, "kafka_manager", manager)
    monkeypatch.setattr(interaction_ingest_api, "system_store", None)
    monkeypatch.setattr(interaction_ingest_api, "feature_store", None)
    monkeypatch.setattr(interaction_ingest_api.time, "time", lambda: now)
    interaction_ingest_api.app.state.runtime.config = SimpleNamespace(
        kafka_config=SimpleNamespace(enable=True)
    )
    return manager


def _sample(name, **labels):
    registry = interaction_ingest_api.app.state.runtime.observability.registry
    return registry.get_sample_value(name, labels) or 0.0


@pytest.mark.asyncio
async def test_interaction_with_fast_client_clock_is_clamped_to_receive_time(monkeypatch):
    received_at = 1_700_000_100.0
    manager = _ingest_with_clock(monkeypatch, received_at)
    clamped_before = _sample(
        "video_commerce_public_event_time_clamped_total",
        kind="interaction",
        bound="upper",
    )
    # Signed buckets: prometheus_client exports no _sum, so read the buckets.
    skew_bucket = "video_commerce_public_event_clock_skew_seconds_bucket"
    le_10_before = _sample(skew_bucket, kind="interaction", le="10.0")
    le_60_before = _sample(skew_bucket, kind="interaction", le="60.0")

    await interaction_ingest_api.ingest_interaction(
        SimpleNamespace(state=SimpleNamespace(request_id="request-fast")),
        UserInteractionRequest(
            user_id="user-1",
            product_id="product-1",
            action=InteractionType.CLICK,
            event_time=received_at + 60.0,
        ),
    )

    assert manager.calls[0]["event_time"] == received_at
    assert manager.calls[0]["client_event_time"] == received_at + 60.0
    assert manager.calls[0]["server_received_at"] == received_at
    assert _sample(
        "video_commerce_public_event_time_clamped_total",
        kind="interaction",
        bound="upper",
    ) == clamped_before + 1
    assert _sample(skew_bucket, kind="interaction", le="10.0") == le_10_before
    assert _sample(skew_bucket, kind="interaction", le="60.0") == le_60_before + 1


@pytest.mark.asyncio
async def test_interaction_with_slow_client_clock_keeps_client_time(monkeypatch):
    # No served-time lookup on the interaction hot path; the PIT lower bound
    # comes from the server available_at instead.
    received_at = 1_700_000_100.0
    manager = _ingest_with_clock(monkeypatch, received_at)

    await interaction_ingest_api.ingest_interaction(
        SimpleNamespace(state=SimpleNamespace(request_id="request-slow")),
        UserInteractionRequest(
            user_id="user-1",
            product_id="product-1",
            action=InteractionType.CLICK,
            event_time=received_at - 60.0,
        ),
    )

    assert manager.calls[0]["event_time"] == received_at - 60.0
    assert manager.calls[0]["client_event_time"] == received_at - 60.0


@pytest.mark.asyncio
async def test_interaction_without_client_time_has_no_client_event_time(monkeypatch):
    manager = _ingest_with_clock(monkeypatch, 1_700_000_100.0)

    await interaction_ingest_api.ingest_interaction(
        SimpleNamespace(state=SimpleNamespace(request_id="request-server-time")),
        UserInteractionRequest(
            user_id="user-1",
            product_id="product-1",
            action=InteractionType.CLICK,
        ),
    )

    assert manager.calls[0]["client_event_time"] is None
    assert manager.calls[0]["event_time"] <= 1_700_000_100.0
