"""
Gateway API for validation, auth, routing, and upload orchestration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import Body, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.middleware import Middleware

from video_commerce.common.auth import AuthValidationError, BearerTokenValidator
from video_commerce.common.cache_codec import json_dumps
from video_commerce.common.config import Config
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.kafka_client import KafkaProducerSupervisor
from video_commerce.common.models import (
    USER_ID_PATTERN,
    RecommendationRequest,
    UserInteractionRequest,
    ViewedImpressionRequest,
)
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.common.service_common import (
    RedisRateLimiter,
    add_inner_http_middleware,
    constant_time_equals,
    parse_trusted_proxy_networks,
    resolve_client_ip,
    build_health_response,
    build_error_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    RoundRobinAsyncClientPool,
)
from video_commerce.data_plane.system_store import (
    ContentIdempotencyKeyInUse,
    PreparedContentTask,
    SystemStore,
    prepare_content_task,
)
from video_commerce.common.telemetry import inject_http_headers

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.:~-]{1,128}$")

app = create_service_app(
    title="Gateway API",
    description="Edge gateway for request validation, auth, routing, and uploads",
    service_name="gateway-api",
)

feature_store: Optional[FeatureStore] = None
system_store: Optional[SystemStore] = None
kafka_manager = None
proxy_client: Optional[httpx.AsyncClient] = None
recommendation_proxy_pool: Optional[RoundRobinAsyncClientPool] = None
interaction_proxy_pool: Optional[RoundRobinAsyncClientPool] = None
rate_limiter: Optional[RedisRateLimiter] = None
failed_auth_limiter: Optional[RedisRateLimiter] = None
trusted_proxy_networks: tuple = ()
object_storage: Optional[ObjectStorage] = None
bearer_token_validator: Optional[BearerTokenValidator] = None
kafka_supervisor: Optional[KafkaProducerSupervisor] = None
# Per-worker cache of public probe payloads: name -> (cached_at, status, content).
_probe_cache: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
_flink_check_cache: Optional[Tuple[float, Dict[str, Any]]] = None
_FLINK_CHECK_CACHE_SECONDS = 15.0
_DOWNSTREAM_PROBE_TIMEOUT_SECONDS = 2.0
# Multipart framing on top of DATA_MAX_FILE_SIZE.
_UPLOAD_BODY_OVERHEAD_BYTES = 1024 * 1024
_UPLOAD_PATH = "/api/content/upload"

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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _rate_limit_key(request: Request, client_ip: str) -> str:
    """Bucket by authenticated principal, falling back to the real client IP.

    Keying on ``request.client.host`` alone put every request behind Caddy in
    one shared bucket.
    """
    claims = getattr(request.state, "auth_claims", None) or {}
    if claims.get("sub"):
        return f"sub:{_digest(str(claims['sub']))}"
    if getattr(request.state, "api_key_authenticated", False):
        return f"key:{_digest(request.headers.get('x-api-key') or '')}"
    return f"ip:{client_ip}"


async def _reject_unauthenticated(
    *,
    client_ip: str,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    request_id_header: str,
):
    """401/503 for a failed credential, or 429 once the client IP keeps failing."""
    if failed_auth_limiter is not None and not await failed_auth_limiter.allow(
        f"ip:{client_ip}"
    ):
        return build_error_response(
            status_code=429,
            code="AUTH_RATE_LIMITED",
            message="Too many failed authentication attempts",
            request_id=request_id,
            headers={request_id_header: request_id},
        )
    return build_error_response(
        status_code=status_code,
        code=code,
        message=message,
        request_id=request_id,
        headers={request_id_header: request_id},
    )


async def auth_and_rate_limit(request: Request, call_next):
    runtime = app.state.runtime
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    request_id_header = (
        runtime.config.monitoring_config.request_id_header
        if runtime.config
        else "X-Request-ID"
    )
    # The shared observability middleware already validated or minted this ID.
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    request.state.request_id = request_id
    security = runtime.config.security_config if runtime.config else None
    auth_mode = security.auth_mode if security else "api_key"
    api_key = runtime.config.api_config.api_key if runtime.config else None
    api_key_enabled = bool(api_key)
    bearer_enabled = bearer_token_validator is not None
    authenticated = auth_mode == "disabled"
    client_ip = resolve_client_ip(request, trusted_proxy_networks)

    bearer_header = request.headers.get("authorization")
    if bearer_token_validator and auth_mode in {"bearer", "api_key_or_bearer"}:
        try:
            claims = await bearer_token_validator.authenticate_async(bearer_header)
            if claims is not None:
                request.state.auth_claims = claims
                authenticated = True
        except AuthValidationError as exc:
            if auth_mode == "bearer" or bearer_header:
                return await _reject_unauthenticated(
                    client_ip=client_ip,
                    status_code=401,
                    code="UNAUTHORIZED",
                    message=str(exc),
                    request_id=request_id,
                    request_id_header=request_id_header,
                )

    if not authenticated and auth_mode in {"api_key", "api_key_or_bearer"} and api_key:
        if constant_time_equals(request.headers.get("x-api-key"), api_key):
            request.state.api_key_authenticated = True
            authenticated = True
        elif auth_mode == "api_key":
            return await _reject_unauthenticated(
                client_ip=client_ip,
                status_code=401,
                code="UNAUTHORIZED",
                message="Invalid API key",
                request_id=request_id,
                request_id_header=request_id_header,
            )

    # Fail closed when the selected mode has no credential configured.
    # Open access must be requested explicitly with SECURITY_AUTH_MODE=disabled.
    if not authenticated and (
        (auth_mode == "api_key" and not api_key_enabled)
        or (auth_mode == "api_key_or_bearer" and not api_key_enabled and not bearer_enabled)
    ):
        return build_error_response(
            status_code=503,
            code="AUTH_NOT_CONFIGURED",
            message=(
                "Authentication is not configured; set API_API_KEY or "
                "SECURITY_AUTH_MODE=disabled for local development"
            ),
            request_id=request_id,
            headers={request_id_header: request_id},
        )

    if not authenticated and auth_mode in {"api_key", "bearer", "api_key_or_bearer"}:
        return await _reject_unauthenticated(
            client_ip=client_ip,
            status_code=401,
            code="UNAUTHORIZED",
            message="Authentication required",
            request_id=request_id,
            request_id_header=request_id_header,
        )

    if rate_limiter and not await rate_limiter.allow(
        _rate_limit_key(request, client_ip)
    ):
        return build_error_response(
            status_code=429,
            code="RATE_LIMITED",
            message="Rate limit exceeded",
            request_id=request_id,
            headers={request_id_header: request_id},
        )

    return await call_next(request)


def _gateway_runtime():
    return app.state.runtime


class _RequestBodyLimitMiddleware:
    """Reject oversized bodies from Content-Length before anything parses them.

    FastAPI spools a whole multipart body to /tmp before the upload handler
    runs, so the handler's own size check came too late to protect the disk.
    Chunked bodies without Content-Length are capped by Caddy's
    ``request_body max_size`` instead.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        config = getattr(_gateway_runtime(), "config", None)
        if (
            scope["type"] != "http"
            or config is None
            or scope.get("method") in {"GET", "HEAD", "OPTIONS"}
        ):
            await self.app(scope, receive, send)
            return
        is_upload = scope.get("path") == _UPLOAD_PATH
        limit = (
            config.data_config.max_file_size + _UPLOAD_BODY_OVERHEAD_BYTES
            if is_upload
            else config.api_config.max_request_size
        )
        content_length = None
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                content_length = value.decode("latin-1").strip()
                break
        error = None
        if content_length is None:
            if is_upload:
                error = (411, "LENGTH_REQUIRED", "Uploads must send Content-Length")
        elif not content_length.isdigit():
            error = (400, "BAD_REQUEST", "Invalid Content-Length")
        elif int(content_length) > limit:
            error = (413, "PAYLOAD_TOO_LARGE", f"Request body exceeds {limit} bytes")
        if error is None:
            await self.app(scope, receive, send)
            return
        request_id = (scope.get("state") or {}).get("request_id") or str(uuid.uuid4())
        response = build_error_response(
            status_code=error[0],
            code=error[1],
            message=error[2],
            request_id=request_id,
            retryable=False,
            headers={config.monitoring_config.request_id_header: request_id},
        )
        await response(scope, receive, send)


add_inner_http_middleware(app, auth_and_rate_limit)
app.user_middleware.append(Middleware(_RequestBodyLimitMiddleware))


def _set_kafka_manager(manager) -> None:
    global kafka_manager
    kafka_manager = manager


@app.on_event("startup")
async def startup_event():
    global feature_store, system_store, kafka_manager, kafka_supervisor, proxy_client, recommendation_proxy_pool, interaction_proxy_pool, rate_limiter, failed_auth_limiter, trusted_proxy_networks, object_storage, bearer_token_validator

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(
        runtime.config.redis_config, runtime.config.cache_config
    )
    await feature_store.initialize()

    if runtime.config.database_config.enable:
        system_store = SystemStore(
            runtime.config.database_config,
            observability=runtime.observability,
        )
        await system_store.initialize()

    object_storage = ObjectStorage(runtime.config.object_storage_config)
    await object_storage.initialize()
    bearer_token_validator = (
        BearerTokenValidator(runtime.config.security_config)
        if runtime.config.security_config.oidc_enabled
        else None
    )
    if (
        runtime.config.security_config.auth_mode == "bearer"
        and bearer_token_validator is None
    ):
        raise ValueError(
            "SECURITY_AUTH_MODE=bearer requires OIDC/JWT validation to be enabled"
        )
    auth_mode = runtime.config.security_config.auth_mode
    if auth_mode in {"api_key", "api_key_or_bearer"} and not (
        runtime.config.api_config.api_key or bearer_token_validator is not None
    ):
        logger.error(
            "SECURITY_AUTH_MODE=%s but no API key or OIDC validator is configured; "
            "protected routes will return 503 AUTH_NOT_CONFIGURED",
            auth_mode,
        )

    if runtime.config.kafka_config.enable:
        kafka_supervisor = KafkaProducerSupervisor(
            runtime.config.kafka_config,
            service_name="gateway-api",
            observability=runtime.observability,
            on_connected=_set_kafka_manager,
        )
        kafka_manager = await kafka_supervisor.start()
    runtime.observability.set_feature_pipeline_mode(
        runtime.config.feature_pipeline_config.mode
    )

    timeout_config = runtime.config.service_topology_config
    proxy_timeout = httpx.Timeout(
        connect=timeout_config.proxy_connect_timeout_seconds,
        read=timeout_config.proxy_read_timeout_seconds,
        write=timeout_config.proxy_write_timeout_seconds,
        pool=timeout_config.proxy_pool_timeout_seconds,
    )
    proxy_limits = httpx.Limits(
        max_connections=timeout_config.proxy_max_connections,
        max_keepalive_connections=timeout_config.proxy_max_keepalive_connections,
        keepalive_expiry=timeout_config.proxy_keepalive_expiry_seconds,
    )
    proxy_client = httpx.AsyncClient(
        timeout=proxy_timeout,
        limits=proxy_limits,
        trust_env=False,
    )
    recommendation_proxy_pool = RoundRobinAsyncClientPool(
        RoundRobinAsyncClientPool.parse_urls(
            timeout_config.recommendation_service_urls,
            fallback=timeout_config.recommendation_service_url,
        ),
        timeout=proxy_timeout,
        limits=proxy_limits,
    )
    interaction_proxy_pool = RoundRobinAsyncClientPool(
        [timeout_config.interaction_ingest_service_url],
        timeout=proxy_timeout,
        limits=proxy_limits,
    )
    api_config = runtime.config.api_config
    trusted_proxy_networks = parse_trusted_proxy_networks(api_config.trusted_proxy_cidrs)
    limiter_timeout = api_config.rate_limit_timeout_ms / 1000.0
    rate_limiter = RedisRateLimiter(
        redis_client=feature_store.redis_client,
        limit=api_config.rate_limit_requests,
        window_seconds=60,
        timeout_seconds=limiter_timeout,
        on_error=runtime.observability.record_rate_limiter_error,
    )
    failed_auth_limiter = RedisRateLimiter(
        redis_client=feature_store.redis_client,
        limit=api_config.unauthenticated_rate_limit_requests,
        window_seconds=60,
        prefix="rate_limit_auth_failures:",
        timeout_seconds=limiter_timeout,
        on_error=runtime.observability.record_rate_limiter_error,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if recommendation_proxy_pool:
        await recommendation_proxy_pool.aclose()
    if interaction_proxy_pool:
        await interaction_proxy_pool.aclose()
    if proxy_client:
        await proxy_client.aclose()
    if feature_store:
        await feature_store.close()
    if system_store:
        await system_store.close()
    if kafka_supervisor:
        await kafka_supervisor.stop()


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
    """Can this gateway pod serve? Only its own state decides.

    Downstream services, Kafka, Postgres and object storage are shared by every
    gateway pod, so failing readiness on them removes all pods at once and
    takes the static edge down with them. Routes that need them return 503 on
    their own; their state is reported by the authenticated /readyz/full.
    """
    return await _cached_probe("readyz", _gateway_own_readiness)


@app.get("/readyz/full")
async def readyz_full():
    return await _gateway_full_readiness()


async def _cached_probe(name: str, builder) -> JSONResponse:
    ttl = app.state.runtime.config.api_config.readiness_cache_seconds
    cached = _probe_cache.get(name)
    now = time.monotonic()
    if cached is not None and now - cached[0] < ttl:
        _, status_code, content = cached
    else:
        built = await builder()
        status_code, content = built.status_code, json.loads(built.body)
        _probe_cache[name] = (now, status_code, content)
    # A fresh response per request: the observability middleware sets the
    # request-id header on it, which must not leak between requests.
    return JSONResponse(status_code=status_code, content=content)


def _auth_check() -> Dict[str, Any]:
    runtime = app.state.runtime
    auth_mode = runtime.config.security_config.auth_mode
    configured = (
        auth_mode == "disabled"
        or bool(runtime.config.api_config.api_key)
        or bearer_token_validator is not None
    )
    return {"status": "healthy" if configured else "unhealthy"}


def _startup_check() -> Dict[str, Any]:
    started = (
        feature_store is not None
        and proxy_client is not None
        and recommendation_proxy_pool is not None
        and interaction_proxy_pool is not None
    )
    return {"status": "healthy" if started else "unhealthy"}


def _record_dependency_statuses(checks: Dict[str, Dict[str, Any]]) -> None:
    observability = app.state.runtime.observability
    for name, check in checks.items():
        observability.set_dependency_status(
            "gateway-api", name, str(check.get("status", "unhealthy"))
        )


async def _gateway_own_readiness() -> JSONResponse:
    checks = {
        "startup": _startup_check(),
        "auth": _auth_check(),
        "redis": await feature_store.health_check()
        if feature_store
        else {"status": "unhealthy"},
    }
    _record_dependency_statuses(checks)
    return build_readiness_response(
        app.state.runtime, checks, critical=checks.keys(), redact_errors=True
    )


def _heartbeat_workers() -> Tuple[str, ...]:
    """Workers expected to heartbeat; feature-worker only runs in python mode.

    Mirrors ``feature_updater.should_run_python_feature_worker``: in flink mode
    the feature-worker exits at startup and never heartbeats.
    """
    mode = str(app.state.runtime.config.feature_pipeline_config.mode or "python").lower()
    if mode == "flink":
        return ("content-worker", "model-trainer")
    return ("content-worker", "feature-worker", "model-trainer")


async def _flink_job_check() -> Optional[Dict[str, Any]]:
    """Informational Flink job state for /readyz/full, cached for 15 seconds."""
    global _flink_check_cache
    pipeline = app.state.runtime.config.feature_pipeline_config
    if str(pipeline.mode).lower() == "python" or not pipeline.flink_rest_url:
        return None
    now = time.monotonic()
    if _flink_check_cache and now - _flink_check_cache[0] < _FLINK_CHECK_CACHE_SECONDS:
        return _flink_check_cache[1]
    from video_commerce.ml.feature_lake_readiness import _flink_status

    try:
        state, checkpoint_ms = await asyncio.wait_for(
            asyncio.to_thread(
                _flink_status, pipeline.flink_rest_url, pipeline.flink_job_name
            ),
            timeout=_DOWNSTREAM_PROBE_TIMEOUT_SECONDS + 1.0,
        )
        check: Dict[str, Any] = {
            "status": "healthy" if state == "RUNNING" else "unhealthy",
            "job_state": state,
            "last_checkpoint_age_seconds": round(time.time() - checkpoint_ms / 1000.0, 1)
            if checkpoint_ms
            else None,
        }
    except Exception as exc:
        check = {"status": "unhealthy", "error": type(exc).__name__}
    _flink_check_cache = (now, check)
    return check


async def _gateway_full_readiness() -> JSONResponse:
    """Authenticated deep readiness: every dependency, strict on unhealthy."""
    runtime = app.state.runtime
    topology = runtime.config.service_topology_config
    (
        recommendation_health,
        ranking_health,
        interaction_health,
    ) = await asyncio.gather(
        _probe_health(f"{topology.recommendation_service_url}/readyz"),
        _probe_health(f"{topology.ranking_service_url}/readyz"),
        _probe_health(f"{topology.interaction_ingest_service_url}/readyz"),
    )
    checks: Dict[str, Dict[str, Any]] = {
        "startup": _startup_check(),
        "auth": _auth_check(),
        "redis": await feature_store.health_check(),
        "database": await _database_check(),
        "kafka": await _kafka_check(),
        "object_storage": await _object_storage_check(),
        "recommendation_service": recommendation_health,
        "ranking_service": ranking_health,
        "interaction_ingest_service": interaction_health,
    }
    if runtime.config.kafka_config.enable:
        for worker_name in _heartbeat_workers():
            checks[worker_name] = await feature_store.get_service_heartbeat_status(
                worker_name
            )
    critical = list(checks)
    flink_check = await _flink_job_check()
    if flink_check is not None:
        checks["flink_feature_job"] = flink_check
    _record_dependency_statuses(checks)
    return build_readiness_response(runtime, checks, critical=critical)


async def _database_check() -> Dict[str, Any]:
    if not system_store:
        return {"status": "healthy", "response_time_ms": 0.0}
    database_status = await system_store.health_check()
    return {
        "status": database_status.status,
        "response_time_ms": database_status.response_time_ms,
        "error": database_status.error,
    }


async def _kafka_check() -> Dict[str, Any]:
    runtime = app.state.runtime
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        return {
            "status": producer_health.get("status", "unhealthy"),
            "response_time_ms": 0.0,
            "error": None
            if producer_health.get("connected")
            else "Kafka producer unavailable",
        }
    if runtime.config.kafka_config.enable:
        return {"status": "unhealthy", "error": "Kafka producer reconnecting"}
    return {"status": "healthy", "response_time_ms": 0.0}


async def _object_storage_check() -> Dict[str, Any]:
    if object_storage is None:
        return {"status": "unhealthy", "error": "Object storage unavailable"}
    return await object_storage.health_check(
        local_path=app.state.runtime.config.data_config.upload_dir
    )


def _bearer_subject(request: Request) -> Optional[str]:
    claims = getattr(request.state, "auth_claims", None)
    if not claims:
        return None
    claim = app.state.runtime.config.security_config.oidc_user_id_claim
    subject = claims.get(claim)
    return str(subject) if subject is not None else None


def _enforce_bearer_user(request: Request, user_id: Optional[str]) -> None:
    """With bearer auth, a request may only act as the token's own user."""
    subject = _bearer_subject(request)
    if subject is None:
        return
    if user_id != subject:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "USER_MISMATCH",
                    "message": "user_id does not match the authenticated subject",
                    "request_id": getattr(request.state, "request_id", None),
                    "retryable": False,
                }
            },
        )


def _has_admin_scope(claims: Dict[str, Any]) -> bool:
    required = app.state.runtime.config.security_config.admin_scope
    scopes = claims.get("scope") or claims.get("scp") or []
    if isinstance(scopes, str):
        scopes = scopes.split()
    return required in {str(scope) for scope in scopes}


@app.post("/api/recommendations")
async def recommendations(request: Request, payload: RecommendationRequest = Body(...)):
    _enforce_bearer_user(request, payload.user_id)
    return await _proxy_json_request(
        proxy_pool=recommendation_proxy_pool,
        path="/api/recommendations",
        target="recommendation-service",
        payload=payload.dict(),
        request=request,
    )


@app.post("/api/interactions")
async def interactions(request: Request, payload: UserInteractionRequest = Body(...)):
    _enforce_bearer_user(request, payload.user_id)
    return await _proxy_json_request(
        proxy_pool=interaction_proxy_pool,
        path="/api/interactions",
        target="interaction-ingest-service",
        payload=payload.dict(),
        request=request,
    )


@app.post("/api/impressions/viewed")
async def viewed_impressions(
    request: Request, payload: ViewedImpressionRequest = Body(...)
):
    return await _proxy_json_request(
        proxy_pool=interaction_proxy_pool,
        path="/api/impressions/viewed",
        target="interaction-ingest-service",
        payload=payload.dict(),
        request=request,
    )


@app.post("/api/content/upload")
async def upload_content(
    request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(
        None, min_length=1, max_length=255, regex=USER_ID_PATTERN
    ),
    priority: str = "normal",
):
    runtime = app.state.runtime
    subject = _bearer_subject(request)
    if subject is not None:
        user_id = user_id or subject
        _enforce_bearer_user(request, user_id)
    if priority not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="Invalid priority")
    idempotency_key = parse_idempotency_key(
        request.headers.get(IDEMPOTENCY_KEY_HEADER)
    )
    if (
        not kafka_manager
        or not runtime.config.kafka_config.enable
        or system_store is None
        or object_storage is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Content uploads require Postgres, object storage, and Kafka",
        )
    dependency_error = await _content_upload_dependency_error(runtime)
    if dependency_error:
        raise HTTPException(status_code=503, detail=dependency_error)

    suffix = validate_upload_file(file, runtime.config)
    hasher = hashlib.sha256() if idempotency_key else None
    file_path, size_bytes = await stream_upload_to_temp_file(
        file=file,
        upload_dir=runtime.config.data_config.upload_dir,
        suffix=suffix,
        max_size=runtime.config.data_config.max_file_size,
        chunk_size=runtime.config.data_config.upload_chunk_size_bytes,
        hasher=hasher,
    )
    request_fingerprint = (
        f"sha256:{hasher.hexdigest()}:{size_bytes}" if hasher is not None else None
    )
    lease_seconds = runtime.config.service_topology_config.content_outbox_lease_seconds
    if idempotency_key:
        try:
            existing = await system_store.get_content_job_by_idempotency_key(
                user_id=user_id, idempotency_key=idempotency_key
            )
        except Exception:
            _remove_staged_file(file_path)
            raise
        if existing is not None:
            _remove_staged_file(file_path)
            return await _idempotent_upload_response(
                existing,
                request_fingerprint=request_fingerprint,
                request=request,
                priority=priority,
                size_bytes=size_bytes,
                lease_seconds=lease_seconds,
            )

    content_id = str(uuid.uuid4())
    storage_path = file_path
    try:
        if object_storage:
            storage_path = await object_storage.persist_staged_file(
                file_path,
                object_name=object_storage.build_object_name(
                    content_id=content_id, suffix=suffix
                ),
                content_type=file.content_type,
            )
            if storage_path != file_path and os.path.exists(file_path):
                os.remove(file_path)
    except Exception:
        _remove_staged_file(file_path)
        raise

    try:
        task, success = await _stage_and_publish_content_task(
            system_store=system_store,
            kafka_manager=kafka_manager,
            content_id=content_id,
            pipeline_version=runtime.config.model_config.content_pipeline_version,
            storage_path=storage_path,
            filename=file.filename,
            user_id=user_id,
            priority=priority,
            request_id=request.state.request_id,
            occurred_at=time.time(),
            lease_seconds=lease_seconds,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
    except ContentIdempotencyKeyInUse:
        # A concurrent request with the same key committed first. Our staging
        # transaction rolled back, so this object is definitely unreferenced.
        await _delete_uploaded_object_quietly(storage_path)
        existing = await system_store.get_content_job_by_idempotency_key(
            user_id=user_id, idempotency_key=idempotency_key
        )
        if existing is None:
            raise HTTPException(
                status_code=409, detail="Idempotency-Key is already in use"
            )
        return await _idempotent_upload_response(
            existing,
            request_fingerprint=request_fingerprint,
            request=request,
            priority=priority,
            size_bytes=size_bytes,
            lease_seconds=lease_seconds,
        )
    except Exception:
        staged_job = await _resolve_unknown_stage_outcome(content_id, storage_path)
        if staged_job is None:
            raise
        # The staging COMMIT succeeded even though the call raised: the job and
        # its outbox row are durable and the publisher will enqueue them.
        success = False
    if not success:
        runtime.observability.record_content_upload(priority, "enqueue_failed")
        return _content_enqueue_pending_response(request, content_id)
    await feature_store.update_content_status(content_id, "queued")
    runtime.observability.record_content_upload(priority, "queued")

    return {
        "content_id": content_id,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "status": "queued",
        "message": f"Content uploaded and queued for processing (priority: {priority}).",
    }


def parse_idempotency_key(value: Optional[str]) -> Optional[str]:
    """Validate an optional Idempotency-Key header (1-128 URL-safe characters)."""
    if value is None:
        return None
    key = value.strip()
    if not _IDEMPOTENCY_KEY_PATTERN.fullmatch(key):
        raise HTTPException(
            status_code=400,
            detail=(
                "Idempotency-Key must be 1-128 characters of "
                "A-Z a-z 0-9 _ . : ~ -"
            ),
        )
    return key


def _remove_staged_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError as exc:
        logger.warning("content_upload_staged_file_cleanup_failed: %s", exc)


async def _delete_uploaded_object_quietly(storage_path: str) -> None:
    try:
        await object_storage.delete_uploaded_object(storage_path)
    except Exception as cleanup_error:
        logger.warning("content_upload_cleanup_failed: %s", cleanup_error)


async def _resolve_unknown_stage_outcome(
    content_id: str, storage_path: str
) -> Optional[Dict[str, Any]]:
    """Decide whether an object is safe to delete after staging raised.

    The staging transaction may have committed even if the call raised (for
    example the connection dropped during COMMIT). Delete the object only when
    the job is definitely absent; if the lookup also fails, keep the object and
    let the orphan reconciler delete it after the grace period if unreferenced.
    """
    try:
        job = await system_store.get_content_job(content_id)
    except Exception as lookup_error:
        logger.warning(
            "content_upload_stage_outcome_unknown: content_id=%s keeping object "
            "for reconciliation: %s",
            content_id,
            lookup_error,
        )
        return None
    if job is None:
        await _delete_uploaded_object_quietly(storage_path)
        return None
    logger.warning(
        "content_upload_staged_despite_error: content_id=%s status=%s",
        content_id,
        job.get("status"),
    )
    return job


def _content_enqueue_pending_response(request: Request, content_id: str) -> Response:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "CONTENT_ENQUEUE_PENDING",
                "message": "Content is durable and pending Kafka acknowledgement",
                "request_id": request.state.request_id,
                "retryable": True,
            },
            "content_id": content_id,
            "status": "pending_publish",
        },
        headers={"Retry-After": "1"},
    )


async def _idempotent_upload_response(
    existing: Dict[str, Any],
    *,
    request_fingerprint: Optional[str],
    request: Request,
    priority: str,
    size_bytes: int,
    lease_seconds: int,
):
    """Replay the original upload result, or reject a reused key with new bytes."""
    runtime = app.state.runtime
    content_id = str(existing["content_id"])
    if existing.get("request_fingerprint") != request_fingerprint:
        runtime.observability.record_content_upload(priority, "idempotency_conflict")
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "IDEMPOTENCY_KEY_REUSED",
                    "message": (
                        "Idempotency-Key was already used for a different upload"
                    ),
                    "request_id": request.state.request_id,
                    "retryable": False,
                },
                "content_id": content_id,
            },
        )
    status = str(existing.get("status") or "")
    if status == "pending_publish" and existing.get("task_event_id"):
        published = await _publish_staged_content_task(
            system_store=system_store,
            kafka_manager=kafka_manager,
            event_id=str(existing["task_event_id"]),
            lease_seconds=lease_seconds,
        )
        if not published:
            runtime.observability.record_content_upload(priority, "enqueue_failed")
            return _content_enqueue_pending_response(request, content_id)
        await feature_store.update_content_status(content_id, "queued")
        status = "queued"
    runtime.observability.record_content_upload(priority, "idempotent_replay")
    return {
        "content_id": content_id,
        "filename": existing.get("filename"),
        "size_bytes": size_bytes,
        "status": status,
        "idempotent_replay": True,
        "message": "Upload already accepted for this Idempotency-Key.",
    }


async def _content_upload_dependency_error(runtime) -> Optional[str]:
    database_status = await system_store.health_check()
    if database_status.status != "healthy":
        return "Content upload database is unavailable"
    kafka_status = await kafka_manager.health_check()
    if not (kafka_status.get("producer") or {}).get("connected"):
        return "Content upload Kafka producer is unavailable"
    storage_status = await object_storage.health_check(
        local_path=runtime.config.data_config.upload_dir
    )
    if storage_status.get("status") != "healthy":
        return "Content upload object storage is unavailable"
    return None


async def _stage_and_publish_content_task(
    *,
    system_store,
    kafka_manager,
    content_id: str,
    pipeline_version: str,
    storage_path: str,
    filename: Optional[str],
    user_id: Optional[str],
    priority: str,
    request_id: Optional[str],
    occurred_at: float,
    lease_seconds: int,
    idempotency_key: Optional[str] = None,
    request_fingerprint: Optional[str] = None,
) -> Tuple[PreparedContentTask, bool]:
    task = prepare_content_task(
        content_id=content_id,
        pipeline_version=pipeline_version,
        storage_path=storage_path,
        filename=filename,
        user_id=user_id,
        priority=priority,
        request_id=request_id,
        timestamp=occurred_at,
    )
    stage_kwargs = {}
    if idempotency_key is not None:
        stage_kwargs = {
            "idempotency_key": idempotency_key,
            "request_fingerprint": request_fingerprint,
        }
    await system_store.stage_content_task(
        task,
        filename=filename,
        storage_path=storage_path,
        user_id=user_id,
        priority=priority,
        request_id=request_id,
        **stage_kwargs,
    )
    published = await _publish_staged_content_task(
        system_store=system_store,
        kafka_manager=kafka_manager,
        event_id=task.event_id,
        lease_seconds=lease_seconds,
    )
    return task, published


async def _publish_staged_content_task(
    *,
    system_store,
    kafka_manager,
    event_id: str,
    lease_seconds: int,
) -> bool:
    """Publish a durable outbox row inline; the publisher retries on failure."""
    worker_id = f"gateway-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        event = await system_store.claim_content_task_outbox_event(
            event_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if event is None:
            return False
        acknowledged = await kafka_manager.publish_video_processing_task_payload(event)
        if not acknowledged:
            raise RuntimeError("Kafka broker did not acknowledge content task")
        marked = await system_store.mark_content_task_outbox_published(
            event_id, worker_id=worker_id
        )
        return bool(marked)
    except Exception as exc:
        try:
            await system_store.mark_content_task_outbox_failed(
                event_id, str(exc), worker_id=worker_id
            )
        except Exception as bookkeeping_error:
            logger.error(
                "content_outbox_failure_record_failed: event_id=%s error=%s",
                event_id,
                bookkeeping_error,
            )
        return False


@app.get("/api/content/{content_id}/status")
async def content_status(content_id: str, request: Request):
    subject = _bearer_subject(request)
    if system_store:
        job = await system_store.get_content_job(content_id)
        if job:
            # 404, not 403, so content IDs of other users cannot be probed.
            if subject is not None and job.get("user_id") != subject:
                raise HTTPException(status_code=404, detail="Content not found")
            return {
                "content_id": content_id,
                "status": job["status"],
                "processed_at": job["updated_at"],
            }

    if subject is not None:
        # The Redis status has no owner, so bearer callers only get DB-backed jobs.
        raise HTTPException(status_code=404, detail="Content not found")
    status = await feature_store.get_content_status(content_id)
    processed_at = await feature_store.get_content_processed_time(content_id)
    if status:
        return {
            "content_id": content_id,
            "status": status,
            "processed_at": processed_at,
        }

    raise HTTPException(status_code=404, detail="Content not found")


@app.get("/api/analytics")
async def analytics(request: Request):
    claims = getattr(request.state, "auth_claims", None)
    if claims is not None and not _has_admin_scope(claims):
        raise HTTPException(status_code=403, detail="Analytics require the admin scope")
    if system_store:
        return await system_store.get_analytics_summary()

    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    return build_error_response(
        status_code=503,
        code="SYSTEM_STORE_UNAVAILABLE",
        message="Analytics require the Postgres-backed system store",
        request_id=request_id,
        headers={
            app.state.runtime.config.monitoring_config.request_id_header: request_id
        },
    )


@app.get("/health")
async def health_check():
    """Public, cached health of this gateway's own dependencies.

    Downstream services and worker heartbeats are reported by the
    authenticated /readyz/full; this endpoint is reachable from the internet
    and must not fan out on every call.
    """
    ttl = app.state.runtime.config.api_config.readiness_cache_seconds
    cached = _probe_cache.get("health")
    now = time.monotonic()
    if cached is not None and now - cached[0] < ttl:
        return cached[2]

    feature_store_health = await feature_store.health_check()
    kafka_health = await _kafka_check()
    if kafka_health.get("status") == "unhealthy":
        kafka_health["status"] = "degraded"
    database_health = await _database_check()
    components = {
        "feature_store": component_health(
            feature_store_health.get("status", "unhealthy"),
            feature_store_health.get("response_time_ms"),
        ),
        "database": component_health(
            database_health.get("status", "unhealthy"),
            database_health.get("response_time_ms"),
        ),
        "kafka": component_health(
            kafka_health.get("status", "healthy"),
            kafka_health.get("response_time_ms"),
        ),
    }
    response = build_health_response(components, app.state.runtime.started_at)
    _probe_cache["health"] = (now, 200, response)
    return response


@app.get("/metrics")
async def metrics():
    worker_statuses = None
    if app.state.runtime.config.kafka_config.enable:
        worker_statuses = {
            worker_name: await feature_store.get_service_heartbeat_status(worker_name)
            for worker_name in _heartbeat_workers()
        }
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
        system_store=system_store,
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
    if content_type not in {
        mime.lower() for mime in config.data_config.allowed_mime_types
    }:
        raise HTTPException(status_code=400, detail="File MIME type not supported")
    return suffix


async def stream_upload_to_temp_file(
    *,
    file: UploadFile,
    upload_dir: str,
    suffix: str,
    max_size: int,
    chunk_size: int,
    hasher: Optional[Any] = None,
) -> Tuple[str, int]:
    """Stream an incoming upload to disk while enforcing a size limit.

    When ``hasher`` (e.g. ``hashlib.sha256()``) is given it is updated with every
    chunk, so the request fingerprint costs no second pass over the file.
    """
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
                    raise HTTPException(
                        status_code=413, detail="Uploaded file exceeds size limit"
                    )
                tmp_file.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
        except Exception:
            tmp_file.close()
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
        finally:
            await file.close()
    return file_path, bytes_written


async def _proxy_json_request(
    proxy_pool: Optional[RoundRobinAsyncClientPool],
    path: str,
    target: str,
    payload: dict,
    request: Request,
) -> Response:
    if proxy_pool is None:
        raise HTTPException(
            status_code=503, detail="Gateway proxy client is unavailable"
        )

    headers = {"Content-Type": "application/json"}
    request_id_header = app.state.runtime.config.monitoring_config.request_id_header
    if request and getattr(request.state, "request_id", None):
        headers[request_id_header] = request.state.request_id

    internal_header = app.state.runtime.config.security_config.internal_service_header
    internal_key = app.state.runtime.config.security_config.internal_service_key
    if internal_key:
        headers[internal_header] = internal_key
    inject_http_headers(headers)

    inflight_recorded = False
    try:
        try:
            content = await request.body()
        except Exception:
            content = json_dumps(payload)
        if not content:
            content = json_dumps(payload)
        app.state.runtime.observability.inc_upstream_inflight(target)
        inflight_recorded = True
        upstream = await proxy_pool.post(path, content=content, headers=headers)
        app.state.runtime.observability.record_upstream_request(
            target, upstream.status_code
        )
    except httpx.TimeoutException as exc:
        app.state.runtime.observability.record_upstream_request(target, "timeout")
        logger.error(f"Gateway proxy timeout: {exc}")
        raise HTTPException(
            status_code=504, detail="Upstream service timed out"
        ) from exc
    except httpx.HTTPError as exc:
        app.state.runtime.observability.record_upstream_request(target, "error")
        logger.error(f"Gateway proxy request failed: {exc}")
        raise HTTPException(
            status_code=502, detail="Upstream service unavailable"
        ) from exc
    finally:
        if inflight_recorded:
            app.state.runtime.observability.dec_upstream_inflight(target)

    content_type = upstream.headers.get("content-type", "application/json")
    response_headers = {}
    if request_id_header in upstream.headers:
        response_headers[request_id_header] = upstream.headers[request_id_header]
    elif request and getattr(request.state, "request_id", None):
        response_headers[request_id_header] = request.state.request_id
    if "retry-after" in upstream.headers:
        response_headers["Retry-After"] = upstream.headers["retry-after"]

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
    inject_http_headers(headers)

    started_at = time.time()
    try:
        response = await asyncio.wait_for(
            proxy_client.get(url, headers=headers),
            timeout=_DOWNSTREAM_PROBE_TIMEOUT_SECONDS,
        )
        payload = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {
            "status": _normalize_probe_status(
                payload.get("status", "healthy" if response.is_success else "degraded")
            ),
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
            "error": payload.get("error", {}).get("message")
            if isinstance(payload.get("error"), dict)
            else None,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "response_time_ms": round((time.time() - started_at) * 1000, 2),
            "error": type(exc).__name__,
        }


def _normalize_probe_status(status: str) -> str:
    if status in {"healthy", "degraded", "unhealthy"}:
        return status
    if status == "ready":
        return "healthy"
    if status == "not_ready":
        return "unhealthy"
    return "healthy"
