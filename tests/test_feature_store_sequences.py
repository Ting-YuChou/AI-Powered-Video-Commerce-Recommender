import json

import pytest

from config import CacheConfig, RedisConfig
from feature_store import FeatureStore


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

    async def execute(self):
        for op in self.ops:
            if op[0] == "lpush":
                _, key, values = op
                bucket = self.client.data.setdefault(key, [])
                for value in values:
                    bucket.insert(0, value)
            elif op[0] == "ltrim":
                _, key, start, end = op
                self.client.data[key] = self.client.data.get(key, [])[start : end + 1]


class FakeRedis:
    def __init__(self):
        self.data = {}

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def lrange(self, key, start, end):
        return self.data.get(key, [])[start : end + 1]


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


@pytest.mark.asyncio
async def test_get_user_sequence_returns_positive_events_oldest_to_newest():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["ui:u1"] = [
        json.dumps({"product_id": "p4", "action": "purchase", "timestamp": 4.0, "occurred_at": 4.0, "event_id": "e4"}),
        "not-json",
        json.dumps({"product_id": "p3", "action": "remove_from_cart", "timestamp": 3.0, "occurred_at": 3.0, "event_id": "e3"}),
        json.dumps({"product_id": "p2", "action": "click", "timestamp": 2.0, "occurred_at": 2.0, "event_id": "e2"}),
        json.dumps({"product_id": "p1", "action": "view", "timestamp": 1.0, "occurred_at": 1.0, "event_id": "e1"}),
    ]

    sequence = await store.get_user_sequence("u1", limit=10)

    assert [event["product_id"] for event in sequence] == ["p1", "p2", "p4"]
    assert [event["event_id"] for event in sequence] == ["e1", "e2", "e4"]


@pytest.mark.asyncio
async def test_user_sequence_token_changes_after_new_positive_event():
    store = FeatureStore(RedisConfig(), CacheConfig())
    fake = FakeRedis()
    store.redis_client = fake
    fake.data["ui:u1"] = [
        json.dumps({"product_id": "p1", "action": "view", "timestamp": 1.0, "occurred_at": 1.0, "event_id": "e1"}),
    ]

    first = await store.get_user_sequence_token("u1")
    fake.data["ui:u1"].insert(
        0,
        json.dumps({"product_id": "p2", "action": "click", "timestamp": 2.0, "occurred_at": 2.0, "event_id": "e2"}),
    )
    second = await store.get_user_sequence_token("u1")

    assert first != second
    assert second["length"] == 2
    assert second["latest_event_id"] == "e2"
