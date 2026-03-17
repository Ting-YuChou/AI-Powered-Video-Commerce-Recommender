"""
Gateway API for validation, auth, routing, and upload orchestration.
"""

from __future__ import annotations

import os
import tempfile
import time
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import Body, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from models import RecommendationRequest, UserInteractionRequest
from service_common import (
    InMemoryRateLimiter,
    build_health_response,
    build_metrics_response,
    component_health,
    configure_service_logging,
    create_service_app,
)

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Gateway API",
    description="Edge gateway for request validation, auth, routing, and uploads",
    service_name="gateway-api",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None
proxy_client: Optional[httpx.AsyncClient] = None
rate_limiter: Optional[InMemoryRateLimiter] = None


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    runtime = app.state.runtime
    api_key = runtime.config.api_config.api_key if runtime.config else None
    if api_key and request.headers.get("x-api-key") != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if rate_limiter:
        client_id = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, kafka_manager, proxy_client, rate_limiter

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

    if runtime.config.kafka_config.enable:
        try:
            kafka_manager = await init_kafka(runtime.config.kafka_config)
        except Exception as exc:
            logger.warning(f"Gateway Kafka init failed: {exc}")
            kafka_manager = None

    proxy_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            timeout=runtime.config.service_topology_config.request_forward_timeout_seconds,
            connect=runtime.config.service_topology_config.proxy_connect_timeout_seconds,
            read=runtime.config.service_topology_config.proxy_read_timeout_seconds,
            write=runtime.config.service_topology_config.proxy_write_timeout_seconds,
            pool=runtime.config.service_topology_config.proxy_pool_timeout_seconds,
        ),
        limits=httpx.Limits(
            max_connections=runtime.config.service_topology_config.proxy_max_connections,
            max_keepalive_connections=runtime.config.service_topology_config.proxy_max_keepalive_connections,
            keepalive_expiry=runtime.config.service_topology_config.proxy_keepalive_expiry_seconds,
        ),
    )
    logger.info(
        "gateway_proxy_client_configured",
        extra={
            "service": runtime.service_name,
            "proxy_max_connections": runtime.config.service_topology_config.proxy_max_connections,
            "proxy_max_keepalive_connections": runtime.config.service_topology_config.proxy_max_keepalive_connections,
            "proxy_keepalive_expiry_seconds": runtime.config.service_topology_config.proxy_keepalive_expiry_seconds,
            "proxy_connect_timeout_seconds": runtime.config.service_topology_config.proxy_connect_timeout_seconds,
            "proxy_read_timeout_seconds": runtime.config.service_topology_config.proxy_read_timeout_seconds,
            "proxy_write_timeout_seconds": runtime.config.service_topology_config.proxy_write_timeout_seconds,
            "proxy_pool_timeout_seconds": runtime.config.service_topology_config.proxy_pool_timeout_seconds,
        },
    )
    rate_limiter = InMemoryRateLimiter(
        limit=runtime.config.api_config.rate_limit_requests,
        window_seconds=60,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if proxy_client:
        await proxy_client.aclose()
    if feature_store:
        await feature_store.close()
    if kafka_manager:
        await close_kafka()


@app.get("/")
async def root():
    return {
        "service": "gateway-api",
        "version": "1.0.0",
        "health": "/health",
    }


@app.post("/api/recommendations")
async def recommendations(payload: RecommendationRequest = Body(...), raw_request: Request = None):
    return await _proxy_json_request(
        url=f"{app.state.runtime.config.service_topology_config.recommendation_service_url}/api/recommendations",
        payload=payload.dict(),
        request=raw_request,
    )


@app.post("/api/interactions")
async def interactions(payload: UserInteractionRequest = Body(...), raw_request: Request = None):
    return await _proxy_json_request(
        url=f"{app.state.runtime.config.service_topology_config.interaction_ingest_service_url}/api/interactions",
        payload=payload.dict(),
        request=raw_request,
    )


@app.post("/api/content/upload")
async def upload_content(
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
    priority: str = "normal",
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    if priority not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="Invalid priority")
    if not kafka_manager or not app.state.runtime.config.kafka_config.enable:
        raise HTTPException(
            status_code=503,
            detail="Content uploads require Kafka-backed worker deployment",
        )

    content_id = f"content_{int(time.time())}_{file.filename}"
    suffix = os.path.splitext(file.filename)[1]
    os.makedirs(app.state.runtime.config.data_config.upload_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=app.state.runtime.config.data_config.upload_dir,
        suffix=suffix,
    ) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        file_path = tmp_file.name

    await feature_store.update_content_status(content_id, "pending")
    success = await kafka_manager.send_video_processing_task(
        content_id=content_id,
        file_path=file_path,
        user_id=user_id,
        priority=priority,
    )
    if not success:
        raise HTTPException(status_code=502, detail="Failed to enqueue video processing task")

    return {
        "content_id": content_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "queued",
        "message": f"Content uploaded and queued for processing (priority: {priority}).",
    }


@app.get("/api/content/{content_id}/status")
async def content_status(content_id: str):
    status = await feature_store.get_content_status(content_id)
    if not status:
        raise HTTPException(status_code=404, detail="Content not found")
    return {
        "content_id": content_id,
        "status": status,
        "processed_at": await feature_store.get_content_processed_time(content_id),
    }


@app.get("/api/analytics")
async def analytics():
    return await feature_store.get_analytics()


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    recommendation_health = await _probe_health(
        f"{app.state.runtime.config.service_topology_config.recommendation_service_url}/health"
    )
    interaction_health = await _probe_health(
        f"{app.state.runtime.config.service_topology_config.interaction_ingest_service_url}/health"
    )
    kafka_health = {"status": "healthy"}
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
            "recommendation_service": component_health(
                recommendation_health.get("status", "unhealthy"),
                recommendation_health.get("response_time_ms"),
                recommendation_health.get("error"),
            ),
            "interaction_ingest_service": component_health(
                interaction_health.get("status", "unhealthy"),
                interaction_health.get("response_time_ms"),
                interaction_health.get("error"),
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


async def _proxy_json_request(url: str, payload: dict, request: Request) -> Response:
    started_at = time.perf_counter()
    upstream_service = urlparse(url).netloc
    runtime = app.state.runtime
    try:
        headers = {"Content-Type": "application/json"}
        request_id_header = runtime.config.monitoring_config.request_id_header
        if request and getattr(request.state, "request_id", None):
            headers[request_id_header] = request.state.request_id
        upstream = await proxy_client.post(url, json=payload, headers=headers)
        upstream_duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        if (
            runtime.config.monitoring_config.enable_profiling_logs
            or upstream_duration_ms >= runtime.config.monitoring_config.profiling_log_min_duration_ms
        ):
            logger.info(
                "gateway_proxy_profile",
                extra={
                    "service": runtime.service_name,
                    "upstream_service": upstream_service,
                    "upstream_status_code": upstream.status_code,
                    "proxy_duration_ms": upstream_duration_ms,
                    "request_path": request.url.path if request else None,
                    "payload_bytes": len(upstream.request.content or b""),
                },
            )
        content_type = upstream.headers.get("content-type", "application/json")
        response_headers = {}
        if request_id_header in upstream.headers:
            response_headers[request_id_header] = upstream.headers[request_id_header]
        if runtime.config.monitoring_config.enable_profiling_logs:
            response_headers["X-Gateway-Proxy-Duration-Ms"] = str(upstream_duration_ms)
            response_headers["X-Upstream-Service"] = upstream_service
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=content_type,
            headers=response_headers,
        )
    except httpx.HTTPError as exc:
        upstream_duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.error(
            "gateway_proxy_request_failed",
            extra={
                "service": runtime.service_name,
                "upstream_service": upstream_service,
                "request_path": request.url.path if request else None,
                "proxy_duration_ms": upstream_duration_ms,
                "exception_type": type(exc).__name__,
                "exception_repr": repr(exc),
                "proxy_max_connections": runtime.config.service_topology_config.proxy_max_connections,
                "proxy_max_keepalive_connections": runtime.config.service_topology_config.proxy_max_keepalive_connections,
                "proxy_pool_timeout_seconds": runtime.config.service_topology_config.proxy_pool_timeout_seconds,
            },
        )
        raise HTTPException(status_code=502, detail=f"Upstream service unavailable: {exc}") from exc


async def _probe_health(url: str) -> dict:
    started_at = time.time()
    try:
        response = await proxy_client.get(url)
        payload = response.json()
        return {
            "status": payload.get("status", "healthy" if response.is_success else "degraded"),
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
            "error": str(exc),
        }
