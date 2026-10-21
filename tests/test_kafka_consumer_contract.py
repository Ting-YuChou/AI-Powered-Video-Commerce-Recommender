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
    config = KafkaConfig(
        consumer_max_poll_interval_ms=600000,
        dead_letter_enable=False,
    )
    client = kafka_client.KafkaConsumerClient(config, group_id="content-processor")

    await client.start(["video-processing-tasks"])

    assert captured["max_poll_interval_ms"] == 600000
    assert client.dlq_producer is None


class _Message:
    topic = "video-processing-tasks"
    partition = 0
    offset = 7
    key = "content-1"
    value = {"content_id": "content-1", "request_id": "request-1"}
    headers = []


class _DeadLetterProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, **kwargs):
        self.sent.append(kwargs)


def _consumer_with_handler(monkeypatch, handler, *, retries=4):
    config = KafkaConfig(
        consumer_handler_retries=retries,
        consumer_handler_retry_backoff_ms=100,
        consumer_handler_retry_max_backoff_ms=250,
        dead_letter_enable=True,
    )
    client = kafka_client.KafkaConsumerClient(config, group_id="content-processor")
    client.register_handler(_Message.topic, handler)
    client.dlq_producer = _DeadLetterProducer()
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(kafka_client.asyncio, "sleep", fake_sleep)
    return client, sleeps


@pytest.mark.asyncio
async def test_non_retryable_failure_goes_straight_to_dlq(monkeypatch):
    calls = []

    async def handler(*_args):
        calls.append(1)
        raise kafka_client.NonRetryableProcessingError("pipeline version mismatch")

    client, sleeps = _consumer_with_handler(monkeypatch, handler)

    assert await client._process_message_with_retries(_Message()) is True

    assert len(calls) == 1
    assert sleeps == []
    [dead_letter] = client.dlq_producer.sent
    assert dead_letter["value"]["error_type"] == "NonRetryableProcessingError"
    assert dead_letter["value"]["source_offset"] == 7


@pytest.mark.asyncio
async def test_retryable_failure_uses_capped_exponential_jittered_backoff(
    monkeypatch,
):
    calls = []

    async def handler(*_args):
        calls.append(1)
        raise RuntimeError("database unavailable")

    client, sleeps = _consumer_with_handler(monkeypatch, handler, retries=4)

    assert await client._process_message_with_retries(_Message()) is True

    assert len(calls) == 4
    assert len(sleeps) == 3
    # Equal jitter: each delay is within [d/2, d] for d = min(250, 100 * 2^n) ms.
    for sleep, ceiling in zip(sleeps, (0.1, 0.2, 0.25)):
        assert ceiling / 2 <= sleep <= ceiling
    assert len(client.dlq_producer.sent) == 1


@pytest.mark.asyncio
async def test_retryable_failure_recovers_without_dlq(monkeypatch):
    calls = []

    async def handler(*_args):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("transient")

    client, sleeps = _consumer_with_handler(monkeypatch, handler)

    assert await client._process_message_with_retries(_Message()) is True
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert client.dlq_producer.sent == []


def test_handler_retry_backoff_bounds():
    backoff = kafka_client.handler_retry_backoff_seconds

    assert backoff(1, base_ms=200, max_ms=10000, jitter=lambda: 0.0) == 0.1
    assert backoff(1, base_ms=200, max_ms=10000, jitter=lambda: 1.0) == 0.2
    assert backoff(4, base_ms=200, max_ms=10000, jitter=lambda: 1.0) == 1.6
    assert backoff(30, base_ms=200, max_ms=1000, jitter=lambda: 1.0) == 1.0
    assert backoff(3, base_ms=0, max_ms=1000) == 0.0
