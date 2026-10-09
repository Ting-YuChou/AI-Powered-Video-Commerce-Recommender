import asyncio
import json
import os
import time
import uuid

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FLINK_CLOSED_LOOP_TESTS") != "1",
    reason="requires the isolated Compose Flink closed-loop stack",
)


def _base_url() -> str:
    return os.environ.get("INTEGRATION_BASE_URL", "http://gateway-api:8000").rstrip("/")


def _headers() -> dict:
    api_key = os.environ.get("API_API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def _postgres_url() -> str:
    value = os.environ["DATABASE_URL"]
    return value.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _eventually(description, assertion, timeout=90.0):
    deadline = time.monotonic() + timeout
    last_value = None
    while time.monotonic() < deadline:
        last_value = await assertion()
        if last_value:
            return last_value
        await asyncio.sleep(0.5)
    pytest.fail(f"timed out waiting for {description}; last value={last_value!r}")


async def _exercise_closed_loop():
    import asyncpg
    import redis.asyncio as redis
    from video_commerce.common.cache_codec import unpack_cache_payload

    event_id = str(uuid.uuid4())
    user_id = f"flink-e2e-user-{event_id}"
    product_id = "prod_1"
    event_time = time.time()
    request_payload = {
        "event_id": event_id,
        "user_id": user_id,
        "product_id": product_id,
        "action": "click",
        "event_time": event_time,
        "context": {"source": "flink-closed-loop"},
    }

    db = await asyncpg.connect(_postgres_url())
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        password=os.environ.get("REDIS_PASSWORD") or None,
        decode_responses=False,
    )
    try:
        async with httpx.AsyncClient(
            base_url=_base_url(),
            headers=_headers(),
            timeout=20.0,
            trust_env=False,
        ) as client:
            accepted = await client.post("/api/interactions", json=request_payload)
            assert accepted.status_code == 202, accepted.text
            assert accepted.json()["event_id"] == event_id
            assert accepted.json()["duplicate"] is False

            async def postgres_received_event():
                count = await db.fetchval(
                    "SELECT count(*) FROM interaction_events WHERE event_id=$1",
                    event_id,
                )
                return count == 1

            await _eventually("one durable interaction row", postgres_received_event)

            async def official_redis_feature_and_sequence():
                feature_raw = await redis_client.get(f"uf:{user_id}")
                sequence_rows = await redis_client.zrange(f"uiz:{user_id}", 0, -1)
                if not feature_raw or not sequence_rows:
                    return False
                features = unpack_cache_payload(feature_raw, "user_features")
                sequence = [json.loads(row) for row in sequence_rows]
                if features.get("total_interactions") != 1:
                    return False
                if not any(row.get("event_id") == event_id for row in sequence):
                    return False
                return {"features": features, "sequence": sequence}

            await _eventually(
                "official Redis user feature and sequence",
                official_redis_feature_and_sequence,
            )

            duplicate = await client.post("/api/interactions", json=request_payload)
            assert duplicate.status_code == 202, duplicate.text
            assert duplicate.json()["event_id"] == event_id
            assert duplicate.json()["duplicate"] is True

            async def still_exactly_once():
                row_count = await db.fetchval(
                    "SELECT count(*) FROM interaction_events WHERE event_id=$1",
                    event_id,
                )
                feature_raw = await redis_client.get(f"uf:{user_id}")
                if not feature_raw:
                    return False
                features = unpack_cache_payload(feature_raw, "user_features")
                return row_count == 1 and features.get("total_interactions") == 1

            await _eventually("idempotent Flink effects", still_exactly_once)

            recommendation = await client.post(
                "/api/recommendations",
                json={
                    "user_id": user_id,
                    "k": 3,
                    "context": {
                        "source": "flink-closed-loop",
                        "session_id": event_id,
                    },
                },
            )
            assert recommendation.status_code == 200, recommendation.text
            metadata = recommendation.json()["metadata"]
            assert metadata["impression_tracking"] == "durable"
            impression_id = metadata["impression_id"]

            async def durable_recommendation_payload():
                payload = await db.fetchval(
                    "SELECT event_payload FROM recommendation_event_outbox "
                    "WHERE impression_id=$1",
                    impression_id,
                )
                if not payload:
                    return False
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return payload

            outbox_payload = await _eventually(
                "next recommendation durable outbox row",
                durable_recommendation_payload,
            )
            snapshot = (outbox_payload.get("metadata") or {}).get(
                "user_feature_snapshot"
            ) or {}
            assert snapshot.get("total_interactions") == 1, outbox_payload
    finally:
        await redis_client.aclose()
        await db.close()


def test_interaction_flows_through_flink_into_next_recommendation():
    asyncio.run(_exercise_closed_loop())
