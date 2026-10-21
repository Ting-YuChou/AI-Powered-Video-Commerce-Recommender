import pytest

from video_commerce.common.config import KafkaConfig
from video_commerce.data_plane.kafka_client import KafkaManager


class RecordingProducer:
    def __init__(self):
        self.calls = []

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        return True


@pytest.mark.asyncio
async def test_recommendation_event_adds_internal_availability_and_lineage(monkeypatch):
    manager = KafkaManager(KafkaConfig())
    manager.producer = RecordingProducer()
    monkeypatch.setattr(
        "video_commerce.data_plane.kafka_client.time.time", lambda: 105.0
    )

    assert await manager.send_recommendation_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=5,
        request_id="request-1",
        metadata={
            "as_of_ts": 100.0,
            "ranking_model_version": "ranking-v1",
            "displayed_items": [{"product_id": "p1"}],
        },
    )

    event = manager.producer.calls[0]["value"]
    assert event["event_time"] == 100.0
    assert event["available_at"] == 105.0
    assert event["source_event_id"] == event["event_id"]
    assert event["source_version"] == "ranking-v1"
    assert event["feature_definition_version"] == "ranking_ltr_v1"
    assert event["payload_schema_version"] == 1
    assert len(event["payload_hash"]) == 64


@pytest.mark.asyncio
async def test_interaction_event_adds_server_owned_availability_and_lineage():
    manager = KafkaManager(KafkaConfig())
    manager.producer = RecordingProducer()

    assert await manager.send_user_interaction(
        user_id="u1",
        product_id="p1",
        action="click",
        context={"surface": "home"},
        event_time=100.0,
        server_received_at=105.0,
    )

    event = manager.producer.calls[0]["value"]
    assert event["event_time"] == 100.0
    assert event["available_at"] == 105.0
    assert event["source_event_id"] == event["event_id"]
    assert event["source_version"] == "interaction-v1"
    assert event["feature_definition_version"] == "ranking_ltr_v1"
    assert len(event["payload_hash"]) == 64


@pytest.mark.asyncio
async def test_interaction_event_keeps_client_time_beside_clamped_event_time():
    manager = KafkaManager(KafkaConfig())
    manager.producer = RecordingProducer()

    assert await manager.send_user_interaction(
        user_id="u1",
        product_id="p1",
        action="click",
        event_time=105.0,
        server_received_at=105.0,
        client_event_time=165.0,
    )

    event = manager.producer.calls[0]["value"]
    assert event["event_time"] == 105.0
    assert event["client_event_time"] == 165.0


@pytest.mark.asyncio
async def test_view_event_anchors_on_clamped_time_and_keeps_client_time(monkeypatch):
    manager = KafkaManager(KafkaConfig())
    manager.producer = RecordingProducer()
    monkeypatch.setattr(
        "video_commerce.data_plane.kafka_client.time.time", lambda: 106.0
    )

    assert await manager.send_recommendation_view(
        event_id="view-1",
        impression_id="imp-1",
        user_id="u1",
        product_id="p1",
        position=1,
        viewed_at=101.0,
        client_viewed_at=41.0,
        served_at=101.0,
        server_received_at=105.0,
    )

    event = manager.producer.calls[0]["value"]
    assert event["event_time"] == 101.0
    assert event["occurred_at"] == 101.0
    assert event["viewed_at"] == 101.0
    assert event["client_viewed_at"] == 41.0
    assert event["served_at"] == 101.0
    assert event["server_received_at"] == 105.0
    assert event["available_at"] == 106.0
