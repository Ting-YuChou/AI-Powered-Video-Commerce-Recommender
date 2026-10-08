"""
Dedicated high-throughput interaction ingest API.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import hashlib
import json
from typing import Optional

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from video_commerce.common.config import Config
from video_commerce.common.event_time import validate_public_event_time
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.kafka_client import close_kafka, init_kafka
from video_commerce.common.models import UserInteractionRequest, ViewedImpressionRequest
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.common.service_common import (
    build_health_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    require_internal_service_auth,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_PENDING_TTL_SECONDS = 30
IDEMPOTENCY_PUBLISHED_TTL_SECONDS = 7 * 24 * 60 * 60

app = create_service_app(
    title="Interaction Ingest Service",
    description="Dedicated interaction ingestion service for async event capture",
    service_name="interaction-ingest-service",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None
system_store: Optional[SystemStore] = None


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    require_internal_service_auth(request, app.state.runtime)
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, kafka_manager, system_store

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

    if runtime.config.database_config.enable:
        system_store = SystemStore(
            runtime.config.database_config, observability=runtime.observability
        )
        await system_store.initialize()

    if runtime.config.kafka_config.enable:
        try:
            kafka_manager = await init_kafka(
                runtime.config.kafka_config,
                observability=runtime.observability,
            )
        except Exception as exc:
            logger.warning(f"Interaction ingest Kafka init failed: {exc}")
            kafka_manager = None


@app.on_event("shutdown")
async def shutdown_event():
    if feature_store:
        await feature_store.close()
    if system_store:
        await system_store.close()
    if kafka_manager:
        await close_kafka()


@app.get("/")
async def root():
    return {
        "service": "interaction-ingest-service",
        "version": "1.0.0",
        "health": "/health",
        "livez": "/livez",
        "readyz": "/readyz",
    }


@app.get("/livez")
async def livez():
    return build_liveness_payload(app.state.runtime)


@app.get("/readyz")
async def readyz():
    runtime = app.state.runtime
    feature_store_health = await feature_store.health_check()
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif runtime.config.kafka_config.enable:
        kafka_health = {"status": "unhealthy", "error": "Kafka producer unavailable"}

    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if runtime.config.database_config.enable:
        if system_store is None:
            database_health = {
                "status": "unhealthy",
                "error": "Postgres idempotency ledger unavailable",
            }
        else:
            database_status = await system_store.health_check()
            database_health = {
                "status": database_status.status,
                "response_time_ms": database_status.response_time_ms,
                "error": database_status.error,
            }

    return build_readiness_response(
        runtime,
        {
            "redis": feature_store_health,
            "kafka": kafka_health,
            "database": database_health,
        },
    )


@app.post("/api/interactions")
async def ingest_interaction(http_request: Request, request: UserInteractionRequest = Body(...)):
    action = request.action.value if hasattr(request.action, "value") else str(request.action)
    server_received_at = time.time()
    try:
        event_time = validate_public_event_time(request.event_time, server_received_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not kafka_manager or not app.state.runtime.config.kafka_config.enable:
        app.state.runtime.observability.record_interaction_ingest(action, "kafka_unavailable")
        raise HTTPException(status_code=503, detail="Interaction ingest requires Kafka to be available")

    event_id = str(request.event_id or uuid.uuid4())
    payload_hash = hashlib.sha256(
        json.dumps(
            {
                "user_id": request.user_id,
                "product_id": request.product_id,
                "action": action,
                "context": request.context,
                "event_time": request.event_time,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    duplicate = False
    owns_claim = False
    claim_token: Optional[str] = None
    uses_postgres_ledger = system_store is not None
    event_payload = {
        "user_id": request.user_id,
        "product_id": request.product_id,
        "action": action,
        "context": request.context,
        "event_time": event_time,
        "server_received_at": server_received_at,
        "request_id": http_request.state.request_id,
    }
    ledger_key = f"interaction:idempotency:{event_id}"
    if uses_postgres_ledger:
        claim = await system_store.claim_interaction_event(
            event_id=event_id,
            payload_hash=payload_hash,
            event_payload=event_payload,
            lease_seconds=IDEMPOTENCY_PENDING_TTL_SECONDS,
        )
        if claim["status"] == "collision":
            raise HTTPException(
                status_code=409,
                detail="event_id was already used for a different interaction",
            )
        if claim["status"] == "pending":
            raise HTTPException(
                status_code=503,
                detail="Interaction with this event_id is still publishing; retry",
                headers={"Retry-After": "1"},
            )
        duplicate = claim["status"] == "duplicate"
        owns_claim = claim["status"] == "claimed"
        claim_token = claim.get("claim_token")
        if owns_claim and not claim_token:
            raise RuntimeError("interaction ledger claim did not return an owner token")
        event_payload = claim.get("event_payload", event_payload)
    elif feature_store and feature_store.redis_client:
        pending = json.dumps({"payload_hash": payload_hash, "status": "pending"})
        owns_claim = bool(
            await feature_store.redis_client.set(
                ledger_key,
                pending,
                nx=True,
                ex=IDEMPOTENCY_PENDING_TTL_SECONDS,
            )
        )
        if not owns_claim:
            for _attempt in range(10):
                existing = await feature_store.redis_client.get(ledger_key)
                if not existing:
                    owns_claim = bool(
                        await feature_store.redis_client.set(
                            ledger_key,
                            pending,
                            nx=True,
                            ex=IDEMPOTENCY_PENDING_TTL_SECONDS,
                        )
                    )
                    if owns_claim:
                        break
                else:
                    record = json.loads(existing)
                    if record.get("payload_hash") != payload_hash:
                        raise HTTPException(
                            status_code=409,
                            detail="event_id was already used for a different interaction",
                        )
                    if record.get("status") == "published":
                        duplicate = True
                        break
                await asyncio.sleep(0.02)
            if not owns_claim and not duplicate:
                raise HTTPException(
                    status_code=503,
                    detail="Interaction with this event_id is still publishing; retry",
                    headers={"Retry-After": "1"},
                )

    success = duplicate or await kafka_manager.send_user_interaction(
        event_id=event_id,
        user_id=event_payload["user_id"],
        product_id=event_payload["product_id"],
        action=event_payload["action"],
        context=event_payload["context"],
        event_time=event_payload["event_time"],
        server_received_at=event_payload["server_received_at"],
        request_id=event_payload["request_id"],
    )
    if not success:
        if owns_claim and uses_postgres_ledger:
            await system_store.release_interaction_event_claim(
                event_id, claim_token=claim_token
            )
        elif owns_claim and feature_store and feature_store.redis_client:
            await feature_store.redis_client.delete(ledger_key)
        app.state.runtime.observability.record_interaction_ingest(action, "publish_failed")
        raise HTTPException(status_code=503, detail="Failed to publish interaction event to Kafka")
    if owns_claim and uses_postgres_ledger:
        marked = await system_store.mark_interaction_event_published(
            event_id, claim_token=claim_token
        )
        if not marked:
            raise HTTPException(
                status_code=503,
                detail="Interaction publish lease expired before acknowledgement",
            )
    elif owns_claim and feature_store and feature_store.redis_client:
        await feature_store.redis_client.set(
            ledger_key,
            json.dumps({"payload_hash": payload_hash, "status": "published"}),
            ex=IDEMPOTENCY_PUBLISHED_TTL_SECONDS,
        )
    app.state.runtime.observability.record_interaction_ingest(action, "accepted")

    if not duplicate:
        asyncio.create_task(
            _invalidate_user_serving_cache(request.user_id),
            name=f"invalidate-serving-cache-{request.user_id}",
        )

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "queue": "kafka",
            "processing_mode": "async",
            "event_id": event_id,
            "duplicate": duplicate,
        },
    )


async def _invalidate_user_serving_cache(user_id: str) -> None:
    try:
        await asyncio.wait_for(
            feature_store.invalidate_user_serving_cache(user_id),
            timeout=0.25,
        )
    except asyncio.TimeoutError:
        logger.warning("serving_cache_invalidation_timed_out", extra={"user_id": user_id})
    except Exception as exc:
        logger.warning(
            "serving_cache_invalidation_failed",
            extra={"user_id": user_id, "error": str(exc)},
        )


@app.post("/api/impressions/viewed")
async def ingest_viewed_impression(
    http_request: Request, request: ViewedImpressionRequest = Body(...)
):
    received_at = time.time()
    try:
        viewed_at = validate_public_event_time(request.viewed_at, received_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not kafka_manager or not app.state.runtime.config.kafka_config.enable:
        raise HTTPException(status_code=503, detail="Viewed impressions require Kafka")

    slate = None
    if system_store is not None:
        slate = await system_store.get_recommendation_impression_slate(
            str(request.impression_id)
        )
        requested_items = {item.product_id: item.position for item in request.items}
        if slate is None or any(
            slate["items"].get(product_id) != position
            for product_id, position in requested_items.items()
        ):
            app.state.runtime.observability.record_recommendation_impression(
                "viewed", "orphan"
            )
            return JSONResponse(
                status_code=202,
                content={"status": "isolated", "reason": "unknown_slate_item"},
            )

    user_id = str(
        (slate or {}).get("user_id")
        or request.context.get("user_id")
        or request.impression_id
    )
    for item in request.items:
        success = await kafka_manager.send_recommendation_view(
            event_id=str(item.event_id),
            impression_id=str(request.impression_id),
            user_id=user_id,
            product_id=item.product_id,
            position=item.position,
            viewed_at=viewed_at,
            context=request.context,
            request_id=http_request.state.request_id,
        )
        if not success:
            raise HTTPException(
                status_code=503, detail="Failed to publish viewed impression"
            )

    if system_store is not None:
        await system_store.record_recommendation_impression_views(
            impression_id=str(request.impression_id),
            viewed_at=viewed_at,
            items=[item.dict() for item in request.items],
            context=request.context,
        )
    app.state.runtime.observability.record_recommendation_impression(
        "viewed", "accepted"
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "impression_id": str(request.impression_id),
            "accepted_items": len(request.items),
        },
    )


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif app.state.runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}

    return build_health_response(
        {
            "feature_store": component_health(
                feature_store_health.get("status", "unhealthy"),
                feature_store_health.get("response_time_ms"),
                feature_store_health.get("error"),
            ),
            "kafka": component_health(
                kafka_health.get("status", "healthy"),
                kafka_health.get("response_time_ms"),
                kafka_health.get("error"),
            ),
        },
        app.state.runtime.started_at,
    )


@app.get("/metrics")
async def metrics():
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
    )
