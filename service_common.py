"""
Shared helpers for multi-service API entrypoints.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import ComponentHealth, HealthResponse, HealthStatus
from observability import (
    ObservabilityManager,
    configure_logging,
    request_id_ctx_var,
)

logger = logging.getLogger(__name__)


class ServiceRuntime:
    """Shared runtime state for service entrypoints."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config = None
        self.observability = ObservabilityManager()
        self.started_at = time.time()
        self.process_id = os.getpid()
        self.active_requests = 0
        self.handled_requests = 0
        self.max_active_requests = 0


class InMemoryRateLimiter:
    """Simple per-key fixed-window rate limiter for the gateway."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"start": 0.0, "count": 0}
        )

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = self._windows[key]
        if now - bucket["start"] >= self.window_seconds:
            bucket["start"] = now
            bucket["count"] = 0

        bucket["count"] += 1
        return bucket["count"] <= self.limit


def create_service_app(
    title: str,
    description: str,
    service_name: str,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
) -> FastAPI:
    """Create a FastAPI app with shared observability middleware."""
    app = FastAPI(
        title=title,
        description=description,
        version="1.0.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
    )
    app.state.runtime = ServiceRuntime(service_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
        runtime.handled_requests += 1
        runtime.max_active_requests = max(runtime.max_active_requests, runtime.active_requests)
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
                    "process_id": runtime.process_id,
                    "worker_active_requests_at_entry": getattr(
                        request.state,
                        "worker_active_requests_at_entry",
                        None,
                    ),
                    "worker_handled_requests": getattr(
                        request.state,
                        "worker_handled_requests",
                        None,
                    ),
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
            if not (runtime.config and not runtime.config.monitoring_config.enable_request_logging):
                logger.info(
                    "request_complete",
                    extra={
                        "method": request.method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                        "client_ip": request.client.host if request.client else None,
                        "service": runtime.service_name,
                        "process_id": runtime.process_id,
                        "worker_active_requests_at_entry": getattr(
                            request.state,
                            "worker_active_requests_at_entry",
                            None,
                        ),
                        "worker_handled_requests": getattr(
                            request.state,
                            "worker_handled_requests",
                            None,
                        ),
                    },
                )
            response.headers[request_id_header] = request_id
            response.headers["X-Service-Process-Pid"] = str(runtime.process_id)
            return response
        finally:
            runtime.observability.http_requests_in_progress.dec()
            runtime.active_requests = max(runtime.active_requests - 1, 0)
            request_id_ctx_var.reset(request_id_token)

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        runtime: ServiceRuntime = app.state.runtime
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        header_name = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found",
                "request_id": request_id,
            },
            headers={header_name: request_id},
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        runtime: ServiceRuntime = app.state.runtime
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        header_name = (
            runtime.config.monitoring_config.request_id_header
            if runtime.config
            else "X-Request-ID"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
            headers={header_name: request_id},
        )

    return app


def configure_service_logging(runtime: ServiceRuntime) -> None:
    """Configure structured logging once config has been loaded."""
    if runtime.config:
        configure_logging(runtime.config.monitoring_config)


async def build_metrics_response(
    runtime: ServiceRuntime,
    feature_store=None,
    kafka_manager=None,
) -> Response:
    """Return Prometheus metrics for the current service."""
    if runtime.config and runtime.config.monitoring_config.enable_prometheus_metrics:
        await runtime.observability.collect_runtime_metrics(
            feature_store=feature_store,
            kafka_manager=kafka_manager,
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
    if any(component.status == HealthStatus.UNHEALTHY for component in components.values()):
        overall = HealthStatus.UNHEALTHY
    elif any(component.status == HealthStatus.DEGRADED for component in components.values()):
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
