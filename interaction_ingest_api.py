"""
Dedicated high-throughput interaction ingest API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from models import UserInteractionRequest
from service_common import (
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

app = create_service_app(
    title="Interaction Ingest Service",
    description="Dedicated interaction ingestion service for async event capture",
    service_name="interaction-ingest-service",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    require_internal_service_auth(request, app.state.runtime)
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, kafka_manager

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

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

    return build_readiness_response(
        runtime,
        {
            "redis": feature_store_health,
            "kafka": kafka_health,
        },
    )


@app.post("/api/interactions")
async def ingest_interaction(http_request: Request, request: UserInteractionRequest = Body(...)):
    action = request.action.value if hasattr(request.action, "value") else str(request.action)

    if not kafka_manager or not app.state.runtime.config.kafka_config.enable:
        app.state.runtime.observability.record_interaction_ingest(action, "kafka_unavailable")
        raise HTTPException(status_code=503, detail="Interaction ingest requires Kafka to be available")

    success = await kafka_manager.send_user_interaction(
        user_id=request.user_id,
        product_id=request.product_id,
        action=action,
        context=request.context,
        timestamp=request.timestamp,
        request_id=http_request.state.request_id,
    )
    if not success:
        app.state.runtime.observability.record_interaction_ingest(action, "publish_failed")
        raise HTTPException(status_code=503, detail="Failed to publish interaction event to Kafka")
    app.state.runtime.observability.record_interaction_ingest(action, "accepted")

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
