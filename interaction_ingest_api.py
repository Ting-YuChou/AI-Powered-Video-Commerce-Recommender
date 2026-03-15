"""
Dedicated high-throughput interaction ingest API.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Body
from fastapi.responses import JSONResponse

from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from models import UserInteractionRequest
from service_common import (
    build_health_response,
    build_metrics_response,
    component_health,
    configure_service_logging,
    create_service_app,
)

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Interaction Ingest Service",
    description="Dedicated interaction ingestion service for async event capture",
    service_name="interaction-ingest-service",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None


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
            kafka_manager = await init_kafka(runtime.config.kafka_config)
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
    }


@app.post("/api/interactions")
async def ingest_interaction(request: UserInteractionRequest = Body(...)):
    action = request.action.value if hasattr(request.action, "value") else str(request.action)

    if kafka_manager and app.state.runtime.config.kafka_config.enable:
        success = await kafka_manager.send_user_interaction_async(
            user_id=request.user_id,
            product_id=request.product_id,
            action=action,
            context=request.context,
            timestamp=request.timestamp,
        )
        if success:
            return JSONResponse(
                status_code=202,
                content={
                    "status": "accepted",
                    "queue": "kafka",
                    "processing_mode": "async",
                },
            )

    stream_id = await feature_store.enqueue_user_interaction(
        user_id=request.user_id,
        product_id=request.product_id,
        action=action,
        context=request.context,
        timestamp=request.timestamp,
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "queue": "redis-stream",
            "stream_id": stream_id.decode() if isinstance(stream_id, bytes) else stream_id,
            "processing_mode": "async",
        },
    )


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_health = await kafka_manager.health_check()
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
