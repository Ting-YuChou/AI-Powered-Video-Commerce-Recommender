"""Stage B WS-2: recommendation hot-swap, outbox budget and readiness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import pytest
from starlette.responses import Response

from video_commerce.common.config import KafkaConfig, RecommendationConfig
from video_commerce.common.models import RecommendationRequest, UserFeatures
from video_commerce.data_plane import kafka_client as kafka_client_module
from video_commerce.data_plane.kafka_client import (
    KafkaMessageTooLargeError,
    KafkaProducerClient,
    KafkaPublishError,
)
from video_commerce.ml.din import DIN_SEQUENCE_CONTEXT_KEY
from video_commerce.ml.ranking_history import RANKING_HISTORY_CONTEXT_KEY
from video_commerce.services.catalog_event_publisher.main import CatalogEventPublisher
from video_commerce.services.recommendation import api as recommendation_api


class RecordingObservability:
    def __init__(self):
        self.impressions = []
        self.terminal = []
        self.catalog_terminal = []
        self.outbox_stats = []

    def record_recommendation(self, **kwargs):
        return None

    def record_recommendation_impression(self, stage, status):
        self.impressions.append((stage, status))

    def record_recommendation_outbox_terminal(self, reason):
        self.terminal.append(reason)

    def record_catalog_outbox_terminal(self, reason):
        self.catalog_terminal.append(reason)

    def update_recommendation_outbox(self, **stats):
        self.outbox_stats.append(stats)


class EventBuildingKafka:
    def __init__(self):
        self.built = []

    def build_recommendation_event(self, **kwargs):
        self.built.append(kwargs)
        return {"event_id": kwargs["event_id"], "metadata": kwargs["metadata"]}


class SlowOutboxStore:
    def __init__(self, delay_seconds=0.5, outcome="inserted"):
        self.delay_seconds = delay_seconds
        self.outcome = outcome
        self.enqueued = []
        self.not_exposed = []

    async def enqueue_recommendation_event(self, event):
        await asyncio.sleep(self.delay_seconds)
        self.enqueued.append(event)
        return self.outcome

    async def mark_recommendation_outbox_not_exposed(self, event_id):
        self.not_exposed.append(event_id)
        return True


def _install_runtime(monkeypatch, *, recommendation_config=None):
    observability = RecordingObservability()
    runtime = SimpleNamespace(
        service_name="recommendation-service",
        active_requests=0,
        handled_requests=0,
        max_active_requests=0,
        observability=observability,
        config=SimpleNamespace(
            recommendation_config=recommendation_config or RecommendationConfig(),
            cache_config=SimpleNamespace(
                hot_path_read_timeout_ms=1000,
                background_write_timeout_ms=1000,
            ),
            monitoring_config=SimpleNamespace(
                enable_profiling_logs=False,
                profiling_log_min_duration_ms=999999,
                request_id_header="X-Request-ID",
            ),
            model_config=SimpleNamespace(ranking_model_path=None),
            vector_config=SimpleNamespace(index_path=None),
            kafka_config=SimpleNamespace(enable=True),
            model_release_config=SimpleNamespace(environment="production"),
            ranking_config=SimpleNamespace(checkpoint_sync_interval_seconds=1),
        ),
    )
    monkeypatch.setattr(
        recommendation_api.app.state, "runtime", runtime, raising=False
    )
    return runtime


# ---------------------------------------------------------------- 58 hot-swap


class PromotingStore:
    def __init__(self):
        self.release_id = "release-a"
        self.generation = 1

    def promote(self, release_id):
        self.release_id = release_id
        self.generation += 1

    async def get_active_model_checkpoint(self, model_name, *, environment):
        assert model_name == recommendation_api.ModelArtifactManager.RANKING_MODEL_NAME
        assert environment == "production"
        return {
            "model_version": f"ranking-{self.release_id}",
            "release_id": self.release_id,
            "generation": self.generation,
            "payload": {
                "model_release_id": self.release_id,
                "active_generation": self.generation,
                "quality_gate_policy_version": "ranking_quality_gate_v4",
            },
        }


class CountingEngine:
    loaded_two_tower_version = "two-tower-1"
    cf_engine = SimpleNamespace(model_version="two-tower-1")

    def __init__(self):
        self.sync_calls = 0

    async def sync_serving_artifacts_if_updated(self):
        self.sync_calls += 1
        return False


def _cache_key(serving_versions):
    context = recommendation_api._build_cache_freshness_context(
        serving_versions,
        {"user": "u1"},
        None,
    )
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()


async def _run_sync_cycles(monkeypatch, runtime, cycles):
    real_sleep = asyncio.sleep
    calls = {"count": 0}

    async def fake_sleep(seconds):
        calls["count"] += 1
        if calls["count"] > cycles:
            raise asyncio.CancelledError()
        await real_sleep(0)

    monkeypatch.setattr(recommendation_api.asyncio, "sleep", fake_sleep)
    await recommendation_api._periodic_serving_artifact_sync(runtime)
    monkeypatch.setattr(recommendation_api.asyncio, "sleep", real_sleep)


@pytest.mark.asyncio
async def test_remote_topology_promote_switches_cache_key_and_lineage_next_cycle(
    monkeypatch,
):
    runtime = _install_runtime(monkeypatch)
    store = PromotingStore()
    engine = CountingEngine()
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(recommendation_api, "ranking_model", None)
    monkeypatch.setattr(
        recommendation_api, "ranking_coordinator_client_pool", object()
    )
    monkeypatch.setattr(recommendation_api, "ranking_client_pool", None)
    monkeypatch.setattr(recommendation_api, "recommendation_engine", engine)
    monkeypatch.setattr(recommendation_api, "vector_search", None)
    monkeypatch.setattr(recommendation_api, "_remote_ranking_release", None)

    await recommendation_api._refresh_remote_ranking_release(runtime)
    before = dict(recommendation_api._refresh_serving_version_context(runtime))
    assert before["model_release_id"] == "release-a"
    assert before["ranking_model"] == "ranking-release-a"

    store.promote("release-b")
    await _run_sync_cycles(monkeypatch, runtime, cycles=1)

    after = recommendation_api._serving_version_context(runtime)
    lineage = recommendation_api._ranking_release_lineage(after)
    assert after["model_release_id"] == "release-b"
    assert after["active_generation"] == 2
    assert lineage["model_release_id"] == "release-b"
    assert lineage["quality_gate_policy_version"] == "ranking_quality_gate_v4"
    assert _cache_key(after) != _cache_key(before)
    # Retrieval artifacts are synced every cycle even without a local ranker.
    assert engine.sync_calls == 1


@pytest.mark.asyncio
async def test_remote_release_lookup_failure_keeps_last_known_release(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    store = PromotingStore()
    engine = CountingEngine()
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(recommendation_api, "ranking_model", None)
    monkeypatch.setattr(recommendation_api, "ranking_client_pool", object())
    monkeypatch.setattr(recommendation_api, "recommendation_engine", engine)
    monkeypatch.setattr(recommendation_api, "vector_search", None)
    monkeypatch.setattr(recommendation_api, "_remote_ranking_release", None)
    await recommendation_api._refresh_remote_ranking_release(runtime)

    async def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "get_active_model_checkpoint", broken)
    await _run_sync_cycles(monkeypatch, runtime, cycles=1)

    context = recommendation_api._serving_version_context(runtime)
    assert context["model_release_id"] == "release-a"
    # A ranking lookup failure must not stall retrieval artifact swaps.
    assert engine.sync_calls == 1


@pytest.mark.asyncio
async def test_local_ranker_metadata_wins_over_remote_pointer(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    monkeypatch.setattr(
        recommendation_api,
        "ranking_model",
        SimpleNamespace(
            model_version="local-ranker",
            artifact_metadata={"model_release_id": "local-release"},
        ),
    )
    monkeypatch.setattr(recommendation_api, "recommendation_engine", None)
    monkeypatch.setattr(recommendation_api, "vector_search", None)
    monkeypatch.setattr(
        recommendation_api, "_remote_ranking_release", {"model_release_id": "x"}
    )

    context = recommendation_api._refresh_serving_version_context(runtime)

    assert context["ranking_model"] == "local-ranker"
    assert context["model_release_id"] == "local-release"


# ------------------------------------------------------- 62 outbox budget


@pytest.mark.asyncio
async def test_slow_outbox_write_times_out_and_late_insert_is_retired(monkeypatch):
    runtime = _install_runtime(
        monkeypatch,
        recommendation_config=RecommendationConfig(impression_outbox_timeout_ms=100),
    )
    store = SlowOutboxStore(delay_seconds=0.5)
    monkeypatch.setattr(recommendation_api, "kafka_manager", EventBuildingKafka())
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(
        recommendation_api, "_impression_outbox_semaphore", asyncio.Semaphore(32)
    )

    started = time.perf_counter()
    durable = await recommendation_api._stage_served_impression_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=5,
        request_id="req-1",
        metadata={"impression_id": "imp-1"},
    )
    elapsed = time.perf_counter() - started

    assert durable is False
    assert elapsed < 0.15
    assert runtime.observability.impressions == [("served", "timeout")]

    await asyncio.sleep(0.6)
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(store.enqueued) == 1
    assert store.not_exposed == [store.enqueued[0]["event_id"]]
    assert runtime.observability.terminal == ["not_exposed"]
    # The permit is released once the shielded write finishes.
    assert recommendation_api._impression_outbox_semaphore._value == 32


@pytest.mark.asyncio
async def test_full_outbox_semaphore_skips_impression_without_queueing(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    store = SlowOutboxStore(delay_seconds=0)
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    monkeypatch.setattr(recommendation_api, "kafka_manager", EventBuildingKafka())
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(recommendation_api, "_impression_outbox_semaphore", semaphore)

    durable = await recommendation_api._stage_served_impression_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=5,
        request_id=None,
        metadata={"impression_id": "imp-2"},
    )

    assert durable is False
    assert store.enqueued == []
    assert runtime.observability.impressions == [("served", "unavailable")]


@pytest.mark.asyncio
async def test_fast_outbox_write_stays_durable_and_wakes_dispatcher(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    wakeup = asyncio.Event()
    monkeypatch.setattr(recommendation_api, "kafka_manager", EventBuildingKafka())
    monkeypatch.setattr(
        recommendation_api, "system_store", SlowOutboxStore(delay_seconds=0)
    )
    monkeypatch.setattr(
        recommendation_api, "_impression_outbox_semaphore", asyncio.Semaphore(2)
    )
    monkeypatch.setattr(recommendation_api, "_recommendation_outbox_wakeup", wakeup)

    durable = await recommendation_api._stage_served_impression_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=5,
        request_id=None,
        metadata={"impression_id": "imp-3"},
    )

    assert durable is True
    assert wakeup.is_set()
    assert runtime.observability.impressions == [("served", "inserted")]


@pytest.mark.asyncio
async def test_cache_hit_response_is_not_held_by_slow_outbox(monkeypatch):
    recommendation_api._recommendation_singleflight.clear()
    runtime = _install_runtime(
        monkeypatch,
        recommendation_config=RecommendationConfig(impression_outbox_timeout_ms=100),
    )
    cached = [
        {
            "product_id": "p1",
            "title": "Product p1",
            "price": 10.0,
            "confidence_score": 0.9,
            "ranking_score": 0.9,
        }
    ]

    class CacheHitFeatureStore:
        async def get_user_serving_context(
            self, user_id, sequence_limit=200, cache_default=False
        ):
            return (
                UserFeatures(user_id=user_id, total_interactions=3),
                recommendation_api._default_user_sequence_token(),
            )

        async def get_user_features(self, user_id, cache_default=False):
            return UserFeatures(user_id=user_id, total_interactions=3)

        async def get_user_sequence_token(self, user_id):
            return recommendation_api._default_user_sequence_token()

        def generate_context_hash(self, context):
            return hashlib.md5(
                json.dumps(context, sort_keys=True, default=str).encode()
            ).hexdigest()

        async def get_cached_recommendations(self, user_id, cache_key):
            return cached

    store = SlowOutboxStore(delay_seconds=0.5)
    monkeypatch.setattr(recommendation_api, "feature_store", CacheHitFeatureStore())
    monkeypatch.setattr(
        recommendation_api,
        "vector_search",
        SimpleNamespace(
            get_catalog_version_context=lambda: {
                "catalog_version": 1,
                "last_updated": 1,
                "product_count": 1,
            }
        ),
    )
    monkeypatch.setattr(recommendation_api, "recommendation_engine", None)
    monkeypatch.setattr(recommendation_api, "ranking_model", None)
    monkeypatch.setattr(recommendation_api, "_remote_ranking_release", None)
    monkeypatch.setattr(recommendation_api, "kafka_manager", EventBuildingKafka())
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(
        recommendation_api, "_impression_outbox_semaphore", asyncio.Semaphore(32)
    )
    recommendation_api._refresh_serving_version_context(runtime)
    request = SimpleNamespace(
        state=SimpleNamespace(
            worker_process_id=1,
            worker_active_requests_at_entry=1,
            worker_handled_requests=1,
            request_started_at=time.perf_counter(),
            request_id="req-cache",
        )
    )

    started = time.perf_counter()
    result = await recommendation_api.get_recommendations(
        request,
        Response(),
        RecommendationRequest(user_id="u1", k=1),
    )
    elapsed = time.perf_counter() - started
    payload = json.loads(result.body)

    assert elapsed < 0.15
    assert payload["metadata"]["cache_hit"] is True
    assert "impression_id" not in payload["metadata"]
    assert payload["metadata"]["impression_tracking"] == "unavailable"
    assert ("served", "timeout") in runtime.observability.impressions
    await asyncio.sleep(0.6)
    for _ in range(5):
        await asyncio.sleep(0)
    assert store.not_exposed


# ------------------------------------------------- 05 feature_context bound


def test_feature_context_drops_history_embeddings_then_din_first():
    context = {
        "device": "mobile",
        RANKING_HISTORY_CONTEXT_KEY: {"embeddings": ["x" * 400] * 400},
        DIN_SEQUENCE_CONTEXT_KEY: {"items": ["p"] * 50},
    }

    bounded, truncated = recommendation_api._bound_feature_context(context, 4096)

    assert truncated == [RANKING_HISTORY_CONTEXT_KEY]
    assert bounded["device"] == "mobile"
    assert DIN_SEQUENCE_CONTEXT_KEY in bounded


def test_feature_context_falls_back_to_bounded_values_when_still_too_large():
    context = {
        "device": "mobile",
        DIN_SEQUENCE_CONTEXT_KEY: {"items": ["p" * 100] * 100},
        "free_text": "y" * 50000,
        "nested": {"a": "z" * 50000},
    }

    bounded, truncated = recommendation_api._bound_feature_context(context, 4096)

    assert len(json.dumps(bounded)) <= 4096
    assert DIN_SEQUENCE_CONTEXT_KEY in truncated
    assert "free_text" in truncated and "nested" in truncated
    assert bounded["device"] == "mobile"
    assert len(bounded["free_text"]) == 512
    assert "nested" not in bounded


def test_small_feature_context_is_untouched():
    context = {"device": "mobile", "_feature_as_of_ts": 1.0}

    assert recommendation_api._bound_feature_context(context, 4096) == (context, [])


@pytest.mark.asyncio
async def test_staged_impression_marks_feature_context_truncated(monkeypatch):
    _install_runtime(
        monkeypatch,
        recommendation_config=RecommendationConfig(feature_context_max_bytes=2048),
    )
    kafka = EventBuildingKafka()
    monkeypatch.setattr(recommendation_api, "kafka_manager", kafka)
    monkeypatch.setattr(
        recommendation_api, "system_store", SlowOutboxStore(delay_seconds=0)
    )
    monkeypatch.setattr(recommendation_api, "_impression_outbox_semaphore", None)

    await recommendation_api._stage_served_impression_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=5,
        request_id=None,
        metadata={
            "impression_id": "imp-4",
            "feature_context": {
                "device": "mobile",
                RANKING_HISTORY_CONTEXT_KEY: {"v": ["x" * 100] * 100},
            },
        },
    )

    metadata = kafka.built[0]["metadata"]
    assert metadata["feature_context_truncated"] is True
    assert metadata["feature_context_truncated_keys"] == [RANKING_HISTORY_CONTEXT_KEY]
    assert RANKING_HISTORY_CONTEXT_KEY not in metadata["feature_context"]


# --------------------------------------------------- 05 terminal dispatch


class OneShotOutboxStore:
    def __init__(self, events):
        self.events = list(events)
        self.failed = []
        self.published = []
        self.claims = 0

    async def get_recommendation_outbox_stats(self):
        return {"pending": len(self.events), "oldest_age_seconds": 0.0}

    async def claim_recommendation_outbox(self, *, worker_id, batch_size, lease_seconds):
        self.claims += 1
        events, self.events = self.events, []
        return events

    async def mark_recommendation_outbox_published(self, event_id, *, worker_id):
        self.published.append(event_id)

    async def mark_recommendation_outbox_failed(self, event_id, error, **kwargs):
        self.failed.append((event_id, error, kwargs))
        if not kwargs["retryable"]:
            return kwargs["terminal_reason"]
        return None


class RaisingKafka:
    def __init__(self, exc):
        self.exc = exc

    async def publish_recommendation_outbox_event(self, event):
        raise self.exc


async def _run_dispatcher_until(store, predicate):
    task = asyncio.create_task(recommendation_api._run_recommendation_outbox_dispatcher())
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if predicate():
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_dispatcher_marks_oversized_event_terminal(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    store = OneShotOutboxStore([{"event_id": "evt-1"}])
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(
        recommendation_api,
        "kafka_manager",
        RaisingKafka(KafkaMessageTooLargeError("too big")),
    )
    monkeypatch.setattr(recommendation_api, "_recommendation_outbox_wakeup", None)

    await _run_dispatcher_until(store, lambda: store.failed)

    event_id, _, kwargs = store.failed[0]
    assert event_id == "evt-1"
    assert kwargs["retryable"] is False
    assert kwargs["terminal_reason"] == "message_too_large"
    assert kwargs["max_attempts"] == 20
    assert runtime.observability.terminal == ["message_too_large"]
    assert ("served", "publish_retry") not in runtime.observability.impressions


@pytest.mark.asyncio
async def test_dispatcher_retries_transient_publish_failures(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    store = OneShotOutboxStore([{"event_id": "evt-2"}])
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(
        recommendation_api, "kafka_manager", RaisingKafka(KafkaPublishError("down"))
    )
    monkeypatch.setattr(recommendation_api, "_recommendation_outbox_wakeup", None)

    await _run_dispatcher_until(store, lambda: store.failed)

    assert store.failed[0][2]["retryable"] is True
    assert runtime.observability.terminal == []
    assert ("served", "publish_retry") in runtime.observability.impressions


@pytest.mark.asyncio
async def test_idle_dispatcher_backs_off_and_reads_stats_on_an_interval(monkeypatch):
    runtime = _install_runtime(monkeypatch)
    store = OneShotOutboxStore([])
    sleeps = []
    real_sleep = asyncio.sleep

    async def fake_wait(timeout):
        sleeps.append(timeout)
        await real_sleep(0)

    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(recommendation_api, "kafka_manager", RaisingKafka(None))
    monkeypatch.setattr(
        recommendation_api, "_wait_for_recommendation_outbox_work", fake_wait
    )

    await _run_dispatcher_until(store, lambda: len(sleeps) >= 6)

    assert sleeps[:5] == [0.25, 0.5, 1.0, 2.0, 2.0]
    assert len(runtime.observability.outbox_stats) == 1


# ---------------------------------------------------------- Kafka producer


class _FakeAioProducer:
    def __init__(self, exc=None):
        self.exc = exc
        self.sent = []

    async def send_and_wait(self, **kwargs):
        if self.exc is not None:
            raise self.exc
        self.sent.append(kwargs)


def _connected_client(exc=None):
    client = KafkaProducerClient(KafkaConfig(enable=True))
    client.producer = _FakeAioProducer(exc)
    client.is_connected = True
    return client


@pytest.mark.asyncio
async def test_send_or_raise_classifies_message_too_large():
    client = _connected_client(kafka_client_module.MessageSizeTooLargeError())

    with pytest.raises(KafkaMessageTooLargeError) as raised:
        await client.send_or_raise("topic", {"a": 1})

    assert raised.value.retryable is False
    assert raised.value.reason == "message_too_large"


@pytest.mark.asyncio
async def test_send_or_raise_marks_other_failures_retryable_and_send_is_unchanged():
    client = _connected_client(RuntimeError("broker down"))

    with pytest.raises(KafkaPublishError) as raised:
        await client.send_or_raise("topic", {"a": 1})
    assert raised.value.retryable is True
    assert not isinstance(raised.value, KafkaMessageTooLargeError)

    too_large = _connected_client(kafka_client_module.MessageSizeTooLargeError())
    assert await too_large.send("topic", {"a": 1}) is False


@pytest.mark.asyncio
async def test_producer_sets_max_request_size_explicitly(monkeypatch):
    captured = {}

    class CapturingProducer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

    monkeypatch.setattr(kafka_client_module, "AIOKafkaProducer", CapturingProducer)
    client = KafkaProducerClient(
        KafkaConfig(enable=True, producer_max_request_size=2_000_000)
    )

    await client.start()

    assert captured["max_request_size"] == 2_000_000


# ----------------------------------------------------- catalog outbox


@pytest.mark.asyncio
async def test_catalog_publisher_marks_oversized_event_terminal():
    calls = []
    observability = RecordingObservability()

    class Store:
        async def claim_catalog_outbox(self, *, worker_id, batch_size, lease_seconds):
            return [{"event_id": "cat-1", "entity_id": "p1"}]

        async def mark_catalog_outbox_failed(self, event_id, error, **kwargs):
            calls.append(kwargs)
            return "message_too_large"

    class Kafka:
        async def send_catalog_feature_event(self, event):
            raise KafkaMessageTooLargeError("too big")

    publisher = CatalogEventPublisher(
        system_store=Store(),
        kafka_manager=Kafka(),
        config=SimpleNamespace(catalog_outbox_batch_size=1, catalog_outbox_lease_seconds=1),
        worker_id="w",
        max_attempts=20,
        observability=observability,
    )

    assert await publisher.publish_once() == 0
    assert calls == [
        {
            "worker_id": "w",
            "retryable": False,
            "terminal_reason": "message_too_large",
            "max_attempts": 20,
        }
    ]
    assert observability.catalog_terminal == ["message_too_large"]


# ------------------------------------------------------------- readiness


class _Health:
    def __init__(self, status):
        self.status = status

    def health_check(self):
        return {"status": self.status}


@pytest.mark.asyncio
async def test_readiness_ignores_remote_ranking_and_kafka_failures(monkeypatch):
    runtime = _install_runtime(monkeypatch)

    class FeatureStore:
        async def health_check(self):
            return {"status": "healthy"}

    async def ranking_down(_runtime):
        return {"status": "unhealthy", "error": "coordinator down"}

    monkeypatch.setattr(recommendation_api, "feature_store", FeatureStore())
    monkeypatch.setattr(recommendation_api, "vector_search", _Health("healthy"))
    monkeypatch.setattr(recommendation_api, "ranking_model", None)
    monkeypatch.setattr(recommendation_api, "system_store", None)
    monkeypatch.setattr(recommendation_api, "kafka_manager", None)
    monkeypatch.setattr(recommendation_api, "ranking_client_pool", object())
    monkeypatch.setattr(
        recommendation_api, "_probe_ranking_service_ready", ranking_down
    )

    response = await recommendation_api.readyz()
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["checks"]["kafka"]["status"] == "degraded"
    assert body["checks"]["ranking_service"]["status"] == "unhealthy"
    assert body["critical"] == ["redis", "vector_search"]
    assert runtime.service_name == body["service"]


@pytest.mark.asyncio
async def test_readiness_fails_without_vector_index(monkeypatch):
    _install_runtime(monkeypatch)

    class FeatureStore:
        async def health_check(self):
            return {"status": "healthy"}

    monkeypatch.setattr(recommendation_api, "feature_store", FeatureStore())
    monkeypatch.setattr(recommendation_api, "vector_search", _Health("unhealthy"))
    monkeypatch.setattr(recommendation_api, "ranking_model", None)
    monkeypatch.setattr(recommendation_api, "system_store", None)
    monkeypatch.setattr(recommendation_api, "kafka_manager", None)
    monkeypatch.setattr(recommendation_api, "ranking_client_pool", None)

    response = await recommendation_api.readyz()

    assert response.status_code == 503
