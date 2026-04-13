"""
Gateway API for validation, auth, routing, and upload orchestration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Optional, Tuple

import httpx
from fastapi import Body, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from auth import AuthValidationError, BearerTokenValidator
from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from models import RecommendationRequest, UserInteractionRequest
from object_storage import ObjectStorage
from service_common import (
    RedisRateLimiter,
    build_health_response,
    build_error_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
)
from system_store import SystemStore

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Gateway API",
    description="Edge gateway for request validation, auth, routing, and uploads",
    service_name="gateway-api",
)

feature_store: Optional[FeatureStore] = None
system_store: Optional[SystemStore] = None
kafka_manager = None
proxy_client: Optional[httpx.AsyncClient] = None
rate_limiter: Optional[RedisRateLimiter] = None
object_storage: Optional[ObjectStorage] = None
bearer_token_validator: Optional[BearerTokenValidator] = None

_AUTH_EXEMPT_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
    "/livez",
    "/readyz",
    "/metrics",
}


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    runtime = app.state.runtime
    if request.url.path not in _AUTH_EXEMPT_PATHS:
        request_id_header = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        request_id = request.headers.get(request_id_header) or str(uuid.uuid4())
        request.state.request_id = request_id
        security = runtime.config.security_config if runtime.config else None
        auth_mode = security.auth_mode if security else "api_key"
        api_key = runtime.config.api_config.api_key if runtime.config else None
        api_key_enabled = bool(api_key)
        bearer_enabled = bearer_token_validator is not None
        authenticated = auth_mode == "disabled"

        bearer_header = request.headers.get("authorization")
        if bearer_token_validator and auth_mode in {"bearer", "api_key_or_bearer"}:
            try:
                claims = bearer_token_validator.authenticate(bearer_header)
                if claims is not None:
                    request.state.auth_claims = claims
                    authenticated = True
            except AuthValidationError as exc:
                if auth_mode == "bearer" or bearer_header:
                    return build_error_response(
                        status_code=401,
                        code="UNAUTHORIZED",
                        message=str(exc),
                        request_id=request_id,
                        headers={request_id_header: request_id},
                    )

        if not authenticated and auth_mode in {"api_key", "api_key_or_bearer"} and api_key:
            if request.headers.get("x-api-key") == api_key:
                authenticated = True
            elif auth_mode == "api_key":
                return build_error_response(
                    status_code=401,
                    code="UNAUTHORIZED",
                    message="Invalid API key",
                    request_id=request_id,
                    headers={request_id_header: request_id},
                )

        if auth_mode == "api_key" and not api_key_enabled:
            authenticated = True
        elif auth_mode == "api_key_or_bearer" and not api_key_enabled and not bearer_enabled:
            authenticated = True

        if not authenticated and auth_mode in {"api_key", "bearer", "api_key_or_bearer"}:
            return build_error_response(
                status_code=401,
                code="UNAUTHORIZED",
                message="Authentication required",
                request_id=request_id,
                headers={request_id_header: request_id},
            )

        if rate_limiter:
            client_id = request.client.host if request.client else "unknown"
            if not await rate_limiter.allow(client_id):
                return build_error_response(
                    status_code=429,
                    code="RATE_LIMITED",
                    message="Rate limit exceeded",
                    request_id=request_id,
                    headers={request_id_header: request_id},
                )

    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, system_store, kafka_manager, proxy_client, rate_limiter, object_storage, bearer_token_validator

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

    if runtime.config.database_config.enable:
        system_store = SystemStore(runtime.config.database_config)
        await system_store.initialize()

    object_storage = ObjectStorage(runtime.config.object_storage_config)
    await object_storage.initialize()
    bearer_token_validator = (
        BearerTokenValidator(runtime.config.security_config)
        if runtime.config.security_config.oidc_enabled
        else None
    )
    if runtime.config.security_config.auth_mode == "bearer" and bearer_token_validator is None:
        raise ValueError("SECURITY_AUTH_MODE=bearer requires OIDC/JWT validation to be enabled")

    if runtime.config.kafka_config.enable:
        try:
            kafka_manager = await init_kafka(runtime.config.kafka_config)
        except Exception as exc:
            logger.warning(f"Gateway Kafka init failed: {exc}")
            kafka_manager = None

    timeout_config = runtime.config.service_topology_config
    proxy_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=timeout_config.proxy_connect_timeout_seconds,
            read=timeout_config.proxy_read_timeout_seconds,
            write=timeout_config.proxy_write_timeout_seconds,
            pool=timeout_config.proxy_pool_timeout_seconds,
        ),
        limits=httpx.Limits(
            max_connections=timeout_config.proxy_max_connections,
            max_keepalive_connections=timeout_config.proxy_max_keepalive_connections,
            keepalive_expiry=timeout_config.proxy_keepalive_expiry_seconds,
        ),
    )
    rate_limiter = RedisRateLimiter(
        redis_client=feature_store.redis_client,
        limit=runtime.config.api_config.rate_limit_requests,
        window_seconds=60,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if proxy_client:
        await proxy_client.aclose()
    if feature_store:
        await feature_store.close()
    if system_store:
        await system_store.close()
    if kafka_manager:
        await close_kafka()


@app.get("/")
async def root():
    return {
        "service": "gateway-api",
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
    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }

    recommendation_health = await _probe_health(
        f"{runtime.config.service_topology_config.recommendation_service_url}/readyz"
    )
    interaction_health = await _probe_health(
        f"{runtime.config.service_topology_config.interaction_ingest_service_url}/readyz"
    )
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "unhealthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif runtime.config.kafka_config.enable:
        kafka_health = {"status": "unhealthy", "error": "Kafka producer unavailable"}

    checks = {
        "redis": feature_store_health,
        "database": database_health,
        "recommendation_service": recommendation_health,
        "interaction_ingest_service": interaction_health,
        "kafka": kafka_health,
    }

    if runtime.config.kafka_config.enable:
        for worker_name in ("content-worker", "feature-worker", "model-trainer"):
            checks[worker_name] = await feature_store.get_service_heartbeat_status(worker_name)

    return build_readiness_response(runtime, checks)


@app.post("/api/recommendations")
async def recommendations(request: Request, payload: RecommendationRequest = Body(...)):
    return await _proxy_json_request(
        url=f"{app.state.runtime.config.service_topology_config.recommendation_service_url}/api/recommendations",
        payload=payload.dict(),
        request=request,
    )


@app.post("/api/interactions")
async def interactions(request: Request, payload: UserInteractionRequest = Body(...)):
    return await _proxy_json_request(
        url=f"{app.state.runtime.config.service_topology_config.interaction_ingest_service_url}/api/interactions",
        payload=payload.dict(),
        request=request,
    )


@app.post("/api/content/upload")
async def upload_content(
    request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
    priority: str = "normal",
):
    runtime = app.state.runtime
    if priority not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="Invalid priority")
    if not kafka_manager or not runtime.config.kafka_config.enable:
        raise HTTPException(
            status_code=503,
            detail="Content uploads require Kafka-backed worker deployment",
        )

    suffix = validate_upload_file(file, runtime.config)
    content_id = str(uuid.uuid4())
    file_path, size_bytes = await stream_upload_to_temp_file(
        file=file,
        upload_dir=runtime.config.data_config.upload_dir,
        suffix=suffix,
        max_size=runtime.config.data_config.max_file_size,
        chunk_size=runtime.config.data_config.upload_chunk_size_bytes,
    )
    storage_path = file_path
    try:
        if object_storage:
            storage_path = await object_storage.persist_staged_file(
                file_path,
                object_name=object_storage.build_object_name(content_id=content_id, suffix=suffix),
                content_type=file.content_type,
            )
            if storage_path != file_path and os.path.exists(file_path):
                os.remove(file_path)
    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    if system_store:
        await system_store.upsert_content_job(
            content_id=content_id,
            filename=file.filename,
            storage_path=storage_path,
            user_id=user_id,
            priority=priority,
            status="pending",
            payload={
                "request_id": request.state.request_id,
                "storage_backend": runtime.config.object_storage_config.backend,
            },
        )

    await feature_store.update_content_status(content_id, "pending")
    success = await kafka_manager.send_video_processing_task(
        content_id=content_id,
        file_path=storage_path,
        filename=file.filename,
        user_id=user_id,
        priority=priority,
        request_id=request.state.request_id,
    )
    if not success:
        if object_storage:
            await object_storage.delete_uploaded_object(storage_path)
        await feature_store.update_content_status(content_id, "failed")
        if system_store:
            await system_store.update_content_job_status(
                content_id,
                "failed",
                error_message="Failed to enqueue video processing task",
                storage_path=storage_path,
            )
        raise HTTPException(status_code=503, detail="Failed to enqueue video processing task")

    return {
        "content_id": content_id,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "status": "queued",
        "message": f"Content uploaded and queued for processing (priority: {priority}).",
    }


@app.get("/api/content/{content_id}/status")
async def content_status(content_id: str):
    status = await feature_store.get_content_status(content_id)
    processed_at = await feature_store.get_content_processed_time(content_id)
    if status:
        return {
            "content_id": content_id,
            "status": status,
            "processed_at": processed_at,
        }

    if system_store:
        job = await system_store.get_content_job(content_id)
        if job:
            return {
                "content_id": content_id,
                "status": job["status"],
                "processed_at": job["updated_at"],
            }

    raise HTTPException(status_code=404, detail="Content not found")


@app.get("/api/analytics")
async def analytics(request: Request):
    if system_store:
        return await system_store.get_analytics_summary()

    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    return build_error_response(
        status_code=503,
        code="SYSTEM_STORE_UNAVAILABLE",
        message="Analytics require the Postgres-backed system store",
        request_id=request_id,
        headers={app.state.runtime.config.monitoring_config.request_id_header: request_id},
    )


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
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif app.state.runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}

    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }

    components = {
        "feature_store": component_health(
            feature_store_health.get("status", "unhealthy"),
            feature_store_health.get("response_time_ms"),
            feature_store_health.get("error"),
        ),
        "database": component_health(
            database_health.get("status", "unhealthy"),
            database_health.get("response_time_ms"),
            database_health.get("error"),
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
    }

    if app.state.runtime.config.kafka_config.enable:
        for worker_name in ("content-worker", "feature-worker", "model-trainer"):
            worker_health = await feature_store.get_service_heartbeat_status(worker_name)
            components[worker_name] = component_health(
                worker_health.get("status", "unhealthy"),
                None,
                worker_health.get("error"),
            )

    return build_health_response(components, app.state.runtime.started_at)


@app.get("/metrics")
async def metrics():
    worker_statuses = None
    if app.state.runtime.config.kafka_config.enable:
        worker_statuses = {
            worker_name: await feature_store.get_service_heartbeat_status(worker_name)
            for worker_name in ("content-worker", "feature-worker", "model-trainer")
        }
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
        worker_statuses=worker_statuses,
    )


def validate_upload_file(file: UploadFile, config: Config) -> str:
    """Validate upload file metadata before streaming it to disk."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if suffix not in {ext.lower() for ext in config.data_config.allowed_extensions}:
        raise HTTPException(status_code=400, detail="File extension not supported")
    if content_type not in {mime.lower() for mime in config.data_config.allowed_mime_types}:
        raise HTTPException(status_code=400, detail="File MIME type not supported")
    return suffix


async def stream_upload_to_temp_file(
    *,
    file: UploadFile,
    upload_dir: str,
    suffix: str,
    max_size: int,
    chunk_size: int,
) -> Tuple[str, int]:
    """Stream an incoming upload to disk while enforcing a size limit."""
    os.makedirs(upload_dir, exist_ok=True)
    bytes_written = 0
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=upload_dir,
        suffix=suffix,
    ) as tmp_file:
        file_path = tmp_file.name
        try:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_size:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds size limit")
                tmp_file.write(chunk)
        except Exception:
            tmp_file.close()
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
        finally:
            await file.close()
    return file_path, bytes_written


async def _proxy_json_request(url: str, payload: dict, request: Request) -> Response:
    if proxy_client is None:
        raise HTTPException(status_code=503, detail="Gateway proxy client is unavailable")

    headers = {"Content-Type": "application/json"}
    request_id_header = app.state.runtime.config.monitoring_config.request_id_header
    if request and getattr(request.state, "request_id", None):
        headers[request_id_header] = request.state.request_id

    internal_header = app.state.runtime.config.security_config.internal_service_header
    internal_key = app.state.runtime.config.security_config.internal_service_key
    if internal_key:
        headers[internal_header] = internal_key

    try:
        upstream = await proxy_client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        logger.error(f"Gateway proxy timeout: {exc}")
        raise HTTPException(status_code=504, detail="Upstream service timed out") from exc
    except httpx.HTTPError as exc:
        logger.error(f"Gateway proxy request failed: {exc}")
        raise HTTPException(status_code=502, detail="Upstream service unavailable") from exc

    content_type = upstream.headers.get("content-type", "application/json")
    response_headers = {}
    if request_id_header in upstream.headers:
        response_headers[request_id_header] = upstream.headers[request_id_header]
    elif request and getattr(request.state, "request_id", None):
        response_headers[request_id_header] = request.state.request_id

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type,
        headers=response_headers,
    )


async def _probe_health(url: str) -> dict:
    if proxy_client is None:
        return {"status": "unhealthy", "error": "Gateway proxy client unavailable"}

    headers = {}
    internal_header = app.state.runtime.config.security_config.internal_service_header
    internal_key = app.state.runtime.config.security_config.internal_service_key
    if internal_key:
        headers[internal_header] = internal_key

    started_at = time.time()
    try:
        response = await proxy_client.get(url, headers=headers)
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "status": _normalize_probe_status(
                payload.get("status", "healthy" if response.is_success else "degraded")
            ),
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
            "error": payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else None,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
            "error": str(exc),
        }


def _normalize_probe_status(status: str) -> str:
    if status in {"healthy", "degraded", "unhealthy"}:
        return status
    if status == "ready":
        return "healthy"
    if status == "not_ready":
        return "unhealthy"
    return "healthy"
