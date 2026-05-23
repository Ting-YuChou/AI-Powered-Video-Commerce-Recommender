"""
OpenTelemetry setup and propagation helpers.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterable, Iterator, List, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

_TRACING_CONFIGURED = False
_FASTAPI_INSTRUMENTED_APPS = set()


def configure_tracing(service_name: str, monitoring_config=None, app=None) -> None:
    """Configure OpenTelemetry SDK and auto-instrument common libraries."""
    if not _tracing_enabled(monitoring_config):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        from opentelemetry.trace import get_tracer_provider
    except Exception as exc:
        logger.warning("OpenTelemetry tracing requested but unavailable: %s", exc)
        return

    global _TRACING_CONFIGURED
    if not _TRACING_CONFIGURED:
        resource = Resource.create(
            {
                "service.name": service_name,
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )
        sample_rate = _sample_rate(monitoring_config)
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(sample_rate)),
        )
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        for instrumentor in (
            HTTPXClientInstrumentor(),
            RedisInstrumentor(),
            SQLAlchemyInstrumentor(),
            LoggingInstrumentor(),
        ):
            try:
                instrumentor.instrument(set_logging_format=False)
            except Exception as exc:
                logger.debug("OpenTelemetry instrumentor skipped: %s", exc)
        _TRACING_CONFIGURED = True

    if app is not None and id(app) not in _FASTAPI_INSTRUMENTED_APPS:
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=get_tracer_provider(),
                excluded_urls="/livez,/readyz,/health,/metrics",
            )
            _FASTAPI_INSTRUMENTED_APPS.add(id(app))
        except Exception as exc:
            logger.debug("FastAPI OpenTelemetry instrumentation skipped: %s", exc)


def inject_trace_headers(headers: Optional[List[Tuple[str, bytes]]] = None) -> List[Tuple[str, bytes]]:
    """Inject W3C trace context into Kafka headers."""
    outgoing = list(headers or [])
    try:
        from opentelemetry.propagate import inject
    except Exception:
        return outgoing

    carrier = {}
    inject(carrier)
    existing = {key for key, _ in outgoing}
    for key, value in carrier.items():
        if key not in existing:
            outgoing.append((key, str(value).encode("utf-8")))
    return outgoing


def inject_http_headers(headers: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Inject W3C trace context into outbound HTTP headers."""
    try:
        from opentelemetry.propagate import inject
    except Exception:
        return headers
    inject(headers)
    return headers


@contextmanager
def http_server_span(
    *,
    method: str,
    path: str,
    headers: Optional[Iterable[Tuple[str, str]]] = None,
) -> Iterator[None]:
    """Create a server span for shared FastAPI middleware."""
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import extract
        from opentelemetry.trace import SpanKind
    except Exception:
        yield
        return

    carrier = {key: value for key, value in headers or []}
    context = extract(carrier)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        f"{method} {path}",
        context=context,
        kind=SpanKind.SERVER,
        attributes={
            "http.request.method": method,
            "url.path": path,
        },
    ):
        yield


@contextmanager
def kafka_consumer_span(
    *,
    topic: str,
    group_id: str,
    headers: Optional[Iterable[Tuple[str, bytes]]],
) -> Iterator[None]:
    """Start a consumer span using trace context from Kafka headers when present."""
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import extract
        from opentelemetry.trace import SpanKind
    except Exception:
        yield
        return

    carrier = {}
    for key, value in headers or []:
        if value is None:
            continue
        carrier[key] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    context = extract(carrier)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        f"kafka consume {topic}",
        context=context,
        kind=SpanKind.CONSUMER,
        attributes={
            "messaging.system": "kafka",
            "messaging.destination.name": topic,
            "messaging.kafka.consumer.group": group_id,
        },
    ):
        yield


def _tracing_enabled(monitoring_config) -> bool:
    raw_env = os.getenv("MONITORING_ENABLE_TRACING")
    if raw_env is not None:
        return raw_env.lower() in {"1", "true", "yes", "on"}
    return bool(getattr(monitoring_config, "enable_tracing", False))


def _sample_rate(monitoring_config) -> float:
    raw_env = os.getenv("MONITORING_TRACING_SAMPLE_RATE")
    raw_value = raw_env if raw_env is not None else getattr(monitoring_config, "tracing_sample_rate", 1.0)
    try:
        return min(1.0, max(0.0, float(raw_value)))
    except (TypeError, ValueError):
        return 1.0
