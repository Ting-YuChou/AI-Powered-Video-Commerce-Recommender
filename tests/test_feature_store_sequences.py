import json

import pytest

from video_commerce.common.cache_codec import pack_cache_payload
from video_commerce.common.config import CacheConfig, RedisConfig
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.common.models import ContentFeatures


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def lpush(self, key, *values):
        self.ops.append(("lpush", key, values))

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", key, start, end))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    def setex(self, key, ttl, value):
        self.ops.append(("setex", key, ttl, value))

    def get(self, key):
        self.ops.append(("get", key))
        self.client.get_calls.append(key)

    def zrange(self, key, start, end):
        self.ops.append(("zrange", key, start, end))

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, mapping))

    def zremrangebyrank(self, key, start, end):
        self.ops.append(("zremrangebyrank", key, start, end))

    async def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "lpush":
                _, key, values = op
                bucket = self.client.data.setdefault(key, [])
                for value in values:
                    bucket.insert(0, value)
            elif op[0] == "ltrim":
                _, key, start, end = op
                self.client.data[key] = self.client.data.get(key, [])[start : end + 1]
            elif op[0] == "setex":
                _, key, _ttl, value = op
                self.client.data[key] = value
            elif op[0] == "get":
                _, key = op
                results.append(self.client.data.get(key))
            elif op[0] == "zrange":
                _, key, start, end = op
                values = self.client.data.get(key, [])
                results.append(values[start:] if end == -1 else values[start : end + 1])
            elif op[0] == "zadd":
                _, key, mapping = op
                bucket = self.client.data.setdefault(key, [])
                bucket.extend(mapping)
            elif op[0] == "zremrangebyrank":
                _, key, start, end = op
                bucket = self.client.data.get(key, [])
                if end < 0:
                    end = len(bucket) + end
                if end >= start:
                    del bucket[start : end + 1]
        return results


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.lrange_calls = []
        self.get_calls = []

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def lrange(self, key, start, end):
        self.lrange_calls.append((key, start, end))
        return self.data.get(key, [])[start : end + 1]

    async def zrange(self, key, start, end):
        values = self.data.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    async def zrevrange(self, key, start, end):
        values = list(reversed(self.data.get(key, [])))
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    async def get(self, key):
        self.get_calls.append(key)
        return self.data.get(key)

    async def mget(self, keys):
        self.get_calls.extend(keys)
        return [self.data.get(key) for key in keys]

    async def scan_iter(self, match=None, count=None):
        prefix = (match or "").rstrip("*")
        for key in list(self.data):
            if not match or key.startswith(prefix):
                yield key

    async def setex(self, key, ttl, value):
        self.data[key] = value


@pytest.mark.asyncio
async def test_log_user_interactions_batch_preserves_sequence_fields():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake

    await store.log_user_interactions_batch(
        "u1",
        [
            {
                "event_id": "e1",
                "schema_version": 1,
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "context": {"page": "home"},
            },
            {
                "event_id": "e2",
                "schema_version": 1,
                "product_id": "p2",
                "action": "click",
                "timestamp": 2.0,
                "occurred_at": 2.0,
                "context": {"page": "home"},
            },
        ],
    )

    newest = json.loads(fake.data["ui:u1"][0])
    assert newest["event_id"] == "e2"
    assert newest["occurred_at"] == 2.0
    assert newest["schema_version"] == 1
    assert len(fake.data["uiza:click:u1"]) == 1
    assert "uiza:view:u1" not in fake.data


@pytest.mark.asyncio
async def test_get_user_sequence_returns_positive_events_oldest_to_newest():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["ui:u1"] = [
        json.dumps(
            {
                "product_id": "p4",
                "action": "purchase",
                "timestamp": 4.0,
                "occurred_at": 4.0,
                "event_id": "e4",
            }
        ),
        "not-json",
        json.dumps(
            {
                "product_id": "p3",
                "action": "remove_from_cart",
                "timestamp": 3.0,
                "occurred_at": 3.0,
                "event_id": "e3",
            }
        ),
        json.dumps(
            {
                "product_id": "p2",
                "action": "click",
                "timestamp": 2.0,
                "occurred_at": 2.0,
                "event_id": "e2",
            }
        ),
        json.dumps(
            {
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "event_id": "e1",
            }
        ),
    ]

    sequence = await store.get_user_sequence("u1", limit=10)

    assert [event["product_id"] for event in sequence] == ["p1", "p2", "p4"]
    assert [event["event_id"] for event in sequence] == ["e1", "e2", "e4"]


@pytest.mark.asyncio
async def test_get_din_behavior_sequences_reads_action_zsets_in_one_pipeline():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["uiza:click:u1"] = [
        json.dumps(
            {
                "product_id": "p1",
                "action": "click",
                "occurred_at": 10.0,
                "available_at": 10.0,
                "event_id": "e1",
            }
        )
    ]
    fake.data["uiza:cart:u1"] = [
        json.dumps(
            {
                "product_id": "p2",
                "action": "add_to_cart",
                "occurred_at": 20.0,
                "available_at": 20.0,
                "event_id": "e2",
            }
        )
    ]

    sequences = await store.get_din_behavior_sequences(
        "u1", as_of_ts=30.0, last_n=2
    )

    assert sequences.actions["click"].product_ids == ("", "p1")
    assert sequences.actions["cart"].product_ids == ("", "p2")
    assert sequences.actions["purchase"].mask == (False, False)


@pytest.mark.asyncio
async def test_user_sequence_token_changes_after_new_positive_event():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["ui:u1"] = [
        json.dumps(
            {
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "event_id": "e1",
            }
        ),
    ]

    first = await store.get_user_sequence_token("u1")
    fake.data["ui:u1"].insert(
        0,
        json.dumps(
            {
                "product_id": "p2",
                "action": "click",
                "timestamp": 2.0,
                "occurred_at": 2.0,
                "event_id": "e2",
            }
        ),
    )
    second = await store.refresh_user_sequence_token("u1")

    assert first != second
    assert second["length"] == 2
    assert second["latest_event_id"] == "e2"


@pytest.mark.asyncio
async def test_user_sequence_token_uses_compact_cached_token_without_lrange():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["ust:200:u1"] = json.dumps(
        {
            "length": 2,
            "latest_event_id": "e2",
            "latest_occurred_at": 2.0,
            "latest_product_id": "p2",
            "latest_action": "click",
        }
    )
    fake.data["ui:u1"] = [
        json.dumps(
            {
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "event_id": "e1",
            }
        ),
    ]

    token = await store.get_user_sequence_token("u1")

    assert token["latest_event_id"] == "e2"
    assert token["latest_product_id"] == "p2"


@pytest.mark.asyncio
async def test_log_user_interactions_batch_refreshes_compact_sequence_token():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake

    await store.log_user_interactions_batch(
        "u1",
        [
            {
                "event_id": "e1",
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
            },
            {
                "event_id": "e2",
                "product_id": "p2",
                "action": "click",
                "timestamp": 2.0,
                "occurred_at": 2.0,
            },
        ],
    )

    token = json.loads(fake.data["ust:200:u1"])
    assert token["length"] == 2
    assert token["latest_event_id"] == "e2"


@pytest.mark.asyncio
async def test_get_user_interactions_reads_flink_zset_newest_first():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["uiz:u1"] = [
        json.dumps(
            {
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "event_id": "e1",
            }
        ),
        json.dumps(
            {
                "product_id": "p2",
                "action": "click",
                "timestamp": 2.0,
                "occurred_at": 2.0,
                "event_id": "e2",
            }
        ),
    ]

    interactions = await store.get_user_interactions("u1", limit=10)

    assert [event["event_id"] for event in interactions] == ["e2", "e1"]


@pytest.mark.asyncio
async def test_get_realtime_window_features_decodes_flink_payloads():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["rtwf:user:u1:5m"] = json.dumps(
        {
            "schema_version": 1,
            "entity_type": "user",
            "entity_id": "u1",
            "window": "5m",
            "views": 10,
            "clicks": 2,
            "add_to_cart": 1,
            "purchases": 1,
            "total_events": 14,
            "click_through_rate": 0.2,
            "conversion_rate": 0.5,
            "window_start": 100.0,
            "window_end": 400.0,
        }
    )

    features = await store.get_realtime_window_features("user", "u1")

    assert features["5m"].views == 10
    assert features["5m"].click_through_rate == 0.2


@pytest.mark.asyncio
async def test_get_realtime_window_features_uses_official_namespace_by_default():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake

    await store.get_realtime_window_features("user", "u1", windows=("5m",))

    assert fake.get_calls == ["rtwf:user:u1:5m"]


@pytest.mark.asyncio
async def test_get_realtime_window_features_can_read_shadow_namespace_explicitly():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake

    await store.get_realtime_window_features(
        "user", "u1", namespace="shadow", windows=("5m",)
    )

    assert fake.get_calls == ["flink:shadow:rtwf:user:u1:5m"]


@pytest.mark.asyncio
async def test_cold_user_serving_context_does_not_fallback_to_lrange():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake

    features, token = await store.get_user_serving_context("cold-user")

    assert features.user_id == "cold-user"
    assert token["length"] == 0
    assert fake.lrange_calls == []


@pytest.mark.asyncio
async def test_unknown_user_snapshot_skips_feature_and_token_reads():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    store._known_user_snapshot_loaded_at = 1.0

    features, token = await store.get_user_serving_context("cold-user")

    assert features.user_id == "cold-user"
    assert token["length"] == 0
    assert fake.get_calls == []
    assert fake.lrange_calls == []


@pytest.mark.asyncio
async def test_warm_user_missing_sequence_token_keeps_lrange_fallback():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    await store._set_user_features(
        "warm-user", store._default_user_features("warm-user")
    )
    fake.data["ui:warm-user"] = [
        json.dumps(
            {
                "product_id": "p1",
                "action": "view",
                "timestamp": 1.0,
                "occurred_at": 1.0,
                "event_id": "e1",
            }
        )
    ]

    _, token = await store.get_user_serving_context("warm-user")

    assert token["length"] == 1
    assert fake.lrange_calls


@pytest.mark.asyncio
async def test_content_feature_snapshot_skips_unknown_content_redis_read():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    store.configure_content_feature_snapshot(enabled=True, max_items=100)
    await store.refresh_content_feature_snapshot()

    result = await store.get_content_features("missing-content")

    assert result is None
    assert "cf:missing-content" not in fake.get_calls


@pytest.mark.asyncio
async def test_content_feature_snapshot_serves_known_content_from_memory():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    features = ContentFeatures(
        content_id="content-1",
        visual_embedding=[0.1, 0.2],
        audio_features={
            "has_audio": True,
            "audio_transcript": "手機 headphones",
            "transcription_status": "completed",
            "speech_categories": ["electronics"],
        },
    )
    fake.data["cf:content-1"] = pack_cache_payload(
        "content_features", features.dict()
    )
    store.configure_content_feature_snapshot(enabled=True, max_items=100)

    count = await store.refresh_content_feature_snapshot()
    fake.get_calls.clear()
    result = await store.get_content_features("content-1")

    assert count == 1
    assert result == features
    assert result.audio_features.audio_transcript == "手機 headphones"
    assert fake.get_calls == []
