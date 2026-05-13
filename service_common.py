"""
Shared helpers for multi-service API entrypoints.
"""

from __future__ import annotations

import os
import random
import time
import uuid
import logging
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import ComponentHealth, HealthResponse, HealthStatus
from observability import (
    ObservabilityManager,
    configure_logging,
    request_id_ctx_var,
)
from telemetry import configure_tracing

logger = logging.getLogger(__name__)


class ServiceRuntime:
    """Shared runtime state for service entrypoints."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.app: FastAPI | None = None
        self.config = None
        self.observability = ObservabilityManager()
        self.started_at = time.time()
        self.process_id = os.getpid()
        self.active_requests = 0
        self.handled_requests = 0
        self.max_active_requests = 0


class RedisRateLimiter:
    """Redis-backed fixed-window rate limiter."""

    def __init__(
        self,
        redis_client,
        limit: int,
        window_seconds: int = 60,
        prefix: str = "rate_limit:",
    ):
        self.redis_client = redis_client
        self.limit = limit
        self.window_seconds = window_seconds
        self.prefix = prefix

    async def allow(self, key: str) -> bool:
        now = int(time.time())
        window = now // self.window_seconds
        redis_key = f"{self.prefix}{window}:{key}"
        pipe = self.redis_client.pipeline(transaction=False)
        pipe.incr(redis_key)
        pipe.expire(redis_key, self.window_seconds)
        current_count, _ = await pipe.execute()
        return int(current_count) <= self.limit


class RoundRobinAsyncClientPool:
    """Small async httpx client pool with per-upstream keepalive state."""

    def __init__(
        self,
        base_urls: Iterable[str],
        *,
        timeout: httpx.Timeout,
        limits: httpx.Limits,
    ) -> None:
        urls = [url.rstrip("/") for url in base_urls if str(url).strip()]
        if not urls:
            raise ValueError("At least one upstream URL is required")
        self.base_urls = urls
        self._clients = [
            httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits)
            for base_url in self.base_urls
        ]
        self._next_index = 0
        self._unhealthy_until = [0.0 for _ in self._clients]
        self._failure_cooldown_seconds = 1.0

    @staticmethod
    def parse_urls(value: str | Iterable[str] | None, fallback: str | None = None) -> List[str]:
        if value is None:
            value = ""
        if isinstance(value, str):
            urls = [item.strip() for item in value.split(",") if item.strip()]
        else:
            urls = [str(item).strip() for item in value if str(item).strip()]
        if not urls and fallback:
            urls = [fallback.strip()]
        return urls

    def _choose_client(self) -> Tuple[int, httpx.AsyncClient]:
        now = time.monotonic()
        fallback_index = self._next_index % len(self._clients)
        for _ in range(len(self._clients)):
            index = self._next_index % len(self._clients)
            self._next_index = (self._next_index + 1) % len(self._clients)
            if self._unhealthy_until[index] <= now:
                return index, self._clients[index]
        self._next_index = (fallback_index + 1) % len(self._clients)
        return fallback_index, self._clients[fallback_index]

    def _mark_unhealthy(self, index: int) -> None:
        self._unhealthy_until[index] = (
            time.monotonic() + self._failure_cooldown_seconds
        )

    def _mark_healthy(self, index: int) -> None:
        self._unhealthy_until[index] = 0.0

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        index, client = self._choose_client()
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.HTTPError:
            self._mark_unhealthy(index)
            raise

        if response.status_code >= 500:
            self._mark_unhealthy(index)
        else:
            self._mark_healthy(index)
        return response

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    retryable: bool | None = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """Build a standardized API error envelope."""
    if retryable is None:
        retryable = status_code >= 500 or status_code == 429
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": retryable,
            }
        },
        headers=headers or {},
    )


def require_internal_service_auth(
    request: Request,
    runtime: ServiceRuntime,
) -> None:
    """Validate internal service auth for private service endpoints."""
    if runtime.config is None:
        return
    path = request.url.path
    if path in {
        "/",
        "/health",
        "/livez",
        "/readyz",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }:
        return
    expected_key = runtime.config.security_config.internal_service_key
    if not expected_key:
        return
    header_name = runtime.config.security_config.internal_service_header
    if request.headers.get(header_name) != expected_key:
        raise HTTPException(status_code=401, detail="Invalid internal service key")


def build_liveness_payload(runtime: ServiceRuntime) -> Dict[str, Any]:
    """Return a simple liveness payload."""
    return {
        "status": "ok",
        "service": runtime.service_name,
        "process_id": runtime.process_id,
        "uptime_seconds": round(time.time() - runtime.started_at, 2),
    }


def build_readiness_response(
    runtime: ServiceRuntime,
    checks: Dict[str, Dict[str, Any]],
) -> JSONResponse:
    """Return a standardized readiness response."""
    ready = all(check.get("status") == "healthy" for check in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "service": runtime.service_name,
            "checks": checks,
            "request_id_header": (
                runtime.config.monitoring_config.request_id_header
                if runtime.config
                else "X-Request-ID"
            ),
        },
    )


def create_service_app(
    title: str,
    description: str,
    service_name: str,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
) -> FastAPI:
    """Create a FastAPI app with shared observability middleware."""
    allow_origins = [
        origin.strip()
        for origin in os.getenv(
            "API_CORS_ORIGINS",
            "http://localhost,http://127.0.0.1,http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]
    app = FastAPI(
        title=title,
        description=description,
        version="1.0.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
    )
    app.state.runtime = ServiceRuntime(service_name)
    app.state.runtime.app = app
    configure_tracing(service_name, app=app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=os.getenv("API_CORS_CREDENTIALS", "true").lower() == "true",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        runtime: ServiceRuntime = app.state.runtime
        request_id_header = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        request_id = request.headers.get(request_id_header) or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.request_started_at = time.perf_counter()
        runtime.active_requests += 1
        runtime.max_active_requests = max(
            runtime.max_active_requests, runtime.active_requests
        )
        request.state.worker_process_id = runtime.process_id
        request.state.worker_active_requests_at_entry = runtime.active_requests
        request.state.worker_handled_requests = runtime.handled_requests
        request_id_token = request_id_ctx_var.set(request_id)
        runtime.observability.http_requests_in_progress.inc()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.perf_counter() - request.state.request_started_at
            path = _get_route_template(request)
            runtime.observability.record_exception(
                request.method,
                path,
                type(exc).__name__,
            )
            runtime.observability.record_request(request.method, path, 500, duration)
            logger.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": path,
                    "status_code": 500,
                    "duration_ms": round(duration * 1000, 2),
                    "client_ip": request.client.host if request.client else None,
                    "service": runtime.service_name,
                },
            )
            raise
        else:
            duration = time.perf_counter() - request.state.request_started_at
            path = _get_route_template(request)
            runtime.observability.record_request(
                request.method,
                path,
                response.status_code,
                duration,
            )
            if _should_log_request(runtime, response.status_code, duration):
                logger.info(
                    "request_complete",
                    extra={
                        "method": request.method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                        "client_ip": request.client.host if request.client else None,
                        "service": runtime.service_name,
                    },
                )
            response.headers[request_id_header] = request_id
            return response
        finally:
            runtime.handled_requests += 1
            runtime.active_requests = max(0, runtime.active_requests - 1)
            runtime.observability.http_requests_in_progress.dec()
            request_id_ctx_var.reset(request_id_token)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        runtime: ServiceRuntime = app.state.runtime
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        header_name = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        detail = (
            exc.detail
            if isinstance(exc.detail, str)
            else HTTPStatus(exc.status_code).phrase
        )
        return build_error_response(
            status_code=exc.status_code,
            code=_error_code_for_status(exc.status_code),
            message=detail,
            request_id=request_id,
            headers={header_name: request_id},
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        runtime: ServiceRuntime = app.state.runtime
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        header_name = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        return build_error_response(
            status_code=500,
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            request_id=request_id,
            headers={header_name: request_id},
        )

    return app


def _should_log_request(
    runtime: ServiceRuntime, status_code: int, duration_seconds: float
) -> bool:
    if not runtime.config:
        return True

    monitoring = runtime.config.monitoring_config
    if not monitoring.enable_request_logging:
        return False

    sample_attr = (
        "error_request_log_sample_rate"
        if status_code >= 400
        else "request_log_sample_rate"
    )
    sample_rate = min(
        1.0, max(0.0, float(getattr(monitoring, sample_attr, 1.0)))
    )
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    return random.random() < sample_rate


def configure_service_logging(runtime: ServiceRuntime) -> None:
    """Configure structured logging once config has been loaded."""
    if runtime.config:
        configure_logging(runtime.config.monitoring_config)
        configure_tracing(
            runtime.service_name,
            runtime.config.monitoring_config,
            app=runtime.app,
        )


async def build_metrics_response(
    runtime: ServiceRuntime,
    feature_store=None,
    kafka_manager=None,
    system_store=None,
    worker_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Response:
    """Return Prometheus metrics for the current service."""
    if runtime.config and runtime.config.monitoring_config.enable_prometheus_metrics:
        await runtime.observability.collect_runtime_metrics(
            feature_store=feature_store,
            kafka_manager=kafka_manager,
            system_store=system_store,
            worker_statuses=worker_statuses,
        )
    return Response(
        content=runtime.observability.prometheus_payload(),
        media_type=runtime.observability.prometheus_content_type,
    )


def component_health(
    status: str,
    response_time_ms: Optional[float] = None,
    error_message: Optional[str] = None,
) -> ComponentHealth:
    """Convert a simple status string into the API health model."""
    mapping = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "unhealthy": HealthStatus.UNHEALTHY,
    }
    return ComponentHealth(
        status=mapping.get(status, HealthStatus.UNHEALTHY),
        response_time_ms=response_time_ms,
        error_message=error_message,
    )


def build_health_response(
    components: Dict[str, ComponentHealth],
    started_at: float,
) -> HealthResponse:
    """Construct a health response with derived overall status."""
    overall = HealthStatus.HEALTHY
    if any(
        component.status == HealthStatus.UNHEALTHY for component in components.values()
    ):
        overall = HealthStatus.UNHEALTHY
    elif any(
        component.status == HealthStatus.DEGRADED for component in components.values()
    ):
        overall = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall,
        components=components,
        timestamp=time.time(),
        uptime_seconds=time.time() - started_at,
    )


def _get_route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    return request.url.path


def _error_code_for_status(status_code: int) -> str:
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "PAYLOAD_TOO_LARGE",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        502: "UPSTREAM_UNAVAILABLE",
        503: "SERVICE_UNAVAILABLE",
        504: "UPSTREAM_TIMEOUT",
    }
    return mapping.get(status_code, f"HTTP_{status_code}")
