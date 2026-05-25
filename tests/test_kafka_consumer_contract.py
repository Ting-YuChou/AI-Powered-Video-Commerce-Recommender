import pytest

from video_commerce.common.config import KafkaConfig
from video_commerce.data_plane import kafka_client


@pytest.mark.asyncio
async def test_consumer_passes_configured_max_poll_interval(monkeypatch):
    captured = {}

    class FakeConsumer:
        def __init__(self, *_topics, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

    monkeypatch.setattr(kafka_client, "AIOKafkaConsumer", FakeConsumer)
    config = KafkaConfig(consumer_max_poll_interval_ms=600000)
    client = kafka_client.KafkaConsumerClient(config, group_id="content-processor")

    await client.start(["video-processing-tasks"])

    assert captured["max_poll_interval_ms"] == 600000
