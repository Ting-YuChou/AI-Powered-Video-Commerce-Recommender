import asyncio

import pytest

from video_commerce.common.cache_codec import unpack_cache_payload
from video_commerce.common.config import CacheConfig, RedisConfig
from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.content_feature_rehydrator import (
    ContentFeatureRehydrator,
)
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.services.recommendation import api as recommendation_api


class TTLRedis:
    """Records SETEX TTLs; ``expire`` simulates Redis evicting a key."""

    def __init__(self):
        self.data = {}
        self.ttls = {}

    async def get(self, key):
        return self.data.get(key)

    async def mget(self, keys):
        return [self.data.get(key) for key in keys]

    async def scan_iter(self, match=None, count=None):
        prefix = (match or "").rstrip("*")
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    async def setex(self, key, ttl, value):
        self.ttls[key] = ttl
        self.data[key] = value

    def expire(self, key):
        self.data.pop(key, None)


class FakeSystemStore:
    def __init__(self, artifacts=None, *, delay=0.0, error=None):
        self.artifacts = dict(artifacts or {})
        self.delay = delay
        self.error = error
        self.calls = []

    async def get_content_feature_artifact(self, content_id, *, hot_path=False):
        self.calls.append((content_id, hot_path))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.artifacts.get(content_id)


class OutcomeSpy:
    def __init__(self):
        self.outcomes = []

    def record_content_feature_rehydration(self, outcome):
        self.outcomes.append(outcome)


def _features(content_id="vid-1"):
    return ContentFeatures(content_id=content_id, visual_embedding=[0.1, 0.2, 0.3])


async def _store_with_expired_projection(features):
    """Worker projects features, Redis expires the key, the snapshot catches up."""
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = TTLRedis()
    store.redis_client = fake
    store.configure_content_feature_snapshot(enabled=True, max_items=100)
    await store.store_content_features(features.content_id, features)
    await store.refresh_content_feature_snapshot()
    fake.expire(f"cf:{features.content_id}")
    await store.refresh_content_feature_snapshot()
    return store, fake


@pytest.mark.asyncio
async def test_redis_projection_alone_loses_features_after_ttl():
    features = _features()
    store, fake = await _store_with_expired_projection(features)

    assert fake.ttls["cf:vid-1"] == CacheConfig().content_features_ttl
    assert await store.get_content_features("vid-1") is None


@pytest.mark.asyncio
async def test_rehydrator_restores_expired_projection_from_postgres():
    features = _features()
    store, fake = await _store_with_expired_projection(features)
    system_store = FakeSystemStore({"vid-1": features.dict()})
    spy = OutcomeSpy()
    rehydrator = ContentFeatureRehydrator(
        feature_store=store, system_store=system_store, observability=spy
    )

    restored = await rehydrator.load("vid-1")

    assert restored == features
    assert system_store.calls == [("vid-1", True)]
    assert spy.outcomes == ["rehydrated"]
    stored = unpack_cache_payload(fake.data["cf:vid-1"], "content_features")
    assert ContentFeatures(**stored) == features
    # The next read is served from the projection without Postgres.
    assert await store.get_content_features("vid-1") == features
    assert len(system_store.calls) == 1


@pytest.mark.asyncio
async def test_rehydrator_shares_one_load_for_concurrent_misses():
    features = _features()
    store, _ = await _store_with_expired_projection(features)
    system_store = FakeSystemStore({"vid-1": features.dict()}, delay=0.05)
    rehydrator = ContentFeatureRehydrator(feature_store=store, system_store=system_store)

    results = await asyncio.gather(*(rehydrator.load("vid-1") for _ in range(5)))

    assert all(result == features for result in results)
    assert len(system_store.calls) == 1


@pytest.mark.asyncio
async def test_rehydrator_remembers_unknown_content_ids():
    store = FeatureStore(RedisConfig(), CacheConfig())
    store.redis_client = TTLRedis()
    system_store = FakeSystemStore()
    spy = OutcomeSpy()
    rehydrator = ContentFeatureRehydrator(
        feature_store=store,
        system_store=system_store,
        negative_ttl_seconds=60.0,
        observability=spy,
    )

    assert await rehydrator.load("unknown") is None
    assert await rehydrator.load("unknown") is None

    assert len(system_store.calls) == 1
    assert spy.outcomes == ["not_found", "negative_cached"]


@pytest.mark.asyncio
async def test_rehydrator_retries_after_read_errors():
    store = FeatureStore(RedisConfig(), CacheConfig())
    store.redis_client = TTLRedis()
    system_store = FakeSystemStore(error=RuntimeError("db down"))
    spy = OutcomeSpy()
    rehydrator = ContentFeatureRehydrator(
        feature_store=store, system_store=system_store, observability=spy
    )

    assert await rehydrator.load("vid-1") is None
    assert await rehydrator.load("vid-1") is None

    assert len(system_store.calls) == 2
    assert spy.outcomes == ["error", "error"]


@pytest.mark.asyncio
async def test_rehydration_finishes_after_the_request_times_out():
    features = _features()
    store, fake = await _store_with_expired_projection(features)
    system_store = FakeSystemStore({"vid-1": features.dict()}, delay=0.05)
    rehydrator = ContentFeatureRehydrator(feature_store=store, system_store=system_store)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(rehydrator.load("vid-1"), timeout=0.01)
    await asyncio.sleep(0.1)

    assert "cf:vid-1" in fake.data
    assert await store.get_content_features("vid-1") == features


@pytest.mark.asyncio
async def test_rehydrator_sheds_new_loads_when_inflight_is_full():
    store = FeatureStore(RedisConfig(), CacheConfig())
    store.redis_client = TTLRedis()
    system_store = FakeSystemStore({}, delay=0.05)
    spy = OutcomeSpy()
    rehydrator = ContentFeatureRehydrator(
        feature_store=store,
        system_store=system_store,
        max_inflight=1,
        observability=spy,
    )

    first = asyncio.create_task(rehydrator.load("a"))
    await asyncio.sleep(0)
    assert await rehydrator.load("b") is None
    await first

    assert [call[0] for call in system_store.calls] == ["a"]
    assert "shed" in spy.outcomes
    await rehydrator.close()


@pytest.mark.asyncio
async def test_recommendation_read_falls_back_to_rehydrator(monkeypatch):
    features = _features()
    store, _ = await _store_with_expired_projection(features)
    rehydrator = ContentFeatureRehydrator(
        feature_store=store,
        system_store=FakeSystemStore({"vid-1": features.dict()}),
    )
    monkeypatch.setattr(recommendation_api, "feature_store", store)
    monkeypatch.setattr(recommendation_api, "content_feature_rehydrator", rehydrator)

    assert await recommendation_api._read_content_features("vid-1") == features


@pytest.mark.asyncio
async def test_recommendation_read_without_rehydrator_keeps_redis_only(monkeypatch):
    features = _features()
    store, _ = await _store_with_expired_projection(features)
    monkeypatch.setattr(recommendation_api, "feature_store", store)
    monkeypatch.setattr(recommendation_api, "content_feature_rehydrator", None)

    assert await recommendation_api._read_content_features("vid-1") is None


class _Closable:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_recommendation_shutdown_closes_rehydrator_and_stores(monkeypatch):
    rehydrator = ContentFeatureRehydrator(
        feature_store=None, system_store=FakeSystemStore()
    )
    store = _Closable()
    system_store = _Closable()

    async def _idle():
        await asyncio.sleep(3600)

    snapshot_task = asyncio.create_task(_idle())
    module_globals = {
        "_impression_outbox_batcher": None,
        "_late_impression_retirer": None,
        "recommendation_outbox_task": None,
        "serving_artifact_sync_task": None,
        "known_user_snapshot_refresh_task": None,
        "content_feature_snapshot_refresh_task": snapshot_task,
        "content_feature_rehydrator": rehydrator,
        "best_effort_task_queue": None,
        "ranking_batcher": None,
        "ranking_coordinator_client_pool": None,
        "ranking_client_pool": None,
        "recommendation_engine": None,
        "feature_store": store,
        "system_store": system_store,
        "kafka_supervisor": None,
        "kafka_manager": None,
    }
    for name, value in module_globals.items():
        monkeypatch.setattr(recommendation_api, name, value)

    await recommendation_api.shutdown_event()

    assert snapshot_task.cancelled()
    assert recommendation_api.content_feature_snapshot_refresh_task is None
    assert recommendation_api.content_feature_rehydrator is None
    assert store.closed and system_store.closed
