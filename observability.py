"""
Observability utilities for request tracing, structured logging, and metrics.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Dict, Optional

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_STANDARD_LOG_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class RequestContextFilter(logging.Filter):
    """Inject request context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_ctx_var.get()
        if not hasattr(record, "service"):
            record.service = "video-commerce-api"
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_ctx_var.get()),
            "service": getattr(record, "service", "video-commerce-api"),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_FIELDS or key in payload:
                continue
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, list, dict)):
                payload[key] = value
            else:
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(monitoring_config) -> None:
    """Configure root logging according to monitoring settings."""
    root_logger = logging.getLogger()
    log_level_name = getattr(monitoring_config, "log_level", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    formatter: logging.Formatter

    if getattr(monitoring_config, "structured_logging", True):
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            getattr(
                monitoring_config,
                "log_format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(log_level)


class ObservabilityManager:
    """Prometheus metrics collector and request instrumentation."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests_total = Counter(
            "video_commerce_http_requests_total",
            "HTTP requests processed by the API",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "video_commerce_http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
            registry=self.registry,
        )
        self.http_requests_in_progress = Gauge(
            "video_commerce_http_requests_in_progress",
            "HTTP requests currently being processed",
            registry=self.registry,
        )
        self.http_request_exceptions_total = Counter(
            "video_commerce_http_request_exceptions_total",
            "Unhandled HTTP request exceptions",
            ["method", "path", "exception_type"],
            registry=self.registry,
        )
        self.process_cpu_percent = Gauge(
            "video_commerce_process_cpu_percent",
            "Process CPU utilization percentage",
            registry=self.registry,
        )
        self.process_resident_memory_bytes = Gauge(
            "video_commerce_process_resident_memory_bytes",
            "Resident memory size in bytes",
            registry=self.registry,
        )
        self.redis_ops_per_sec = Gauge(
            "video_commerce_redis_ops_per_sec",
            "Redis instantaneous operations per second",
            registry=self.registry,
        )
        self.redis_connected_clients = Gauge(
            "video_commerce_redis_connected_clients",
            "Redis connected clients",
            registry=self.registry,
        )
        self.redis_used_memory_bytes = Gauge(
            "video_commerce_redis_used_memory_bytes",
            "Redis used memory in bytes",
            registry=self.registry,
        )
        self.kafka_producer_connected = Gauge(
            "video_commerce_kafka_producer_connected",
            "Kafka producer connection status",
            registry=self.registry,
        )
        self.kafka_consumer_lag = Gauge(
            "video_commerce_kafka_consumer_lag",
            "Kafka consumer lag by consumer group (-1 when unavailable)",
            ["group_id"],
            registry=self.registry,
        )
        self.worker_live_instances = Gauge(
            "video_commerce_worker_live_instances",
            "Number of live worker instances observed via Redis heartbeats",
            ["service"],
            registry=self.registry,
        )
        self.worker_status = Gauge(
            "video_commerce_worker_status",
            "Worker health status observed via Redis heartbeats (1=healthy, 0=unhealthy)",
            ["service"],
            registry=self.registry,
        )
        self._process = psutil.Process()

    def record_request(self, method: str, path: str, status_code: int, duration: float) -> None:
        status = str(status_code)
        self.http_requests_total.labels(method=method, path=path, status=status).inc()
        self.http_request_duration_seconds.labels(method=method, path=path).observe(duration)

    def record_exception(self, method: str, path: str, exception_type: str) -> None:
        self.http_request_exceptions_total.labels(
            method=method,
            path=path,
            exception_type=exception_type,
        ).inc()

    async def collect_runtime_metrics(
        self,
        feature_store=None,
        kafka_manager=None,
        worker_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Refresh gauges from the current process and backing services."""
        self.process_cpu_percent.set(self._process.cpu_percent())
        self.process_resident_memory_bytes.set(self._process.memory_info().rss)

        if feature_store and getattr(feature_store, "redis_client", None):
            try:
                redis_info = await feature_store.redis_client.info()
                self.redis_ops_per_sec.set(redis_info.get("instantaneous_ops_per_sec", 0))
                self.redis_connected_clients.set(redis_info.get("connected_clients", 0))
                self.redis_used_memory_bytes.set(redis_info.get("used_memory", 0))
            except Exception:
                pass

        if kafka_manager:
            producer_connected = 1 if kafka_manager.producer.is_connected else 0
            self.kafka_producer_connected.set(producer_connected)
            await self._collect_kafka_consumer_lag(kafka_manager)
        else:
            self.kafka_producer_connected.set(0)
            self.kafka_consumer_lag.labels(group_id="none").set(-1)

        if worker_statuses:
            for service_name, status in worker_statuses.items():
                self.worker_live_instances.labels(service=service_name).set(
                    status.get("live_instances", 0)
                )
                self.worker_status.labels(service=service_name).set(
                    1 if status.get("status") == "healthy" else 0
                )

    async def _collect_kafka_consumer_lag(self, kafka_manager) -> None:
        if not kafka_manager._consumers:
            self.kafka_consumer_lag.labels(group_id="none").set(-1)
            return

        try:
            self.kafka_consumer_lag.remove("none")
        except KeyError:
            pass
        for group_id, consumer_client in kafka_manager._consumers.items():
            lag_value = -1
            try:
                consumer = consumer_client.consumer
                if consumer_client.is_connected and consumer is not None:
                    assignment = consumer.assignment()
                    if assignment:
                        end_offsets = await consumer.end_offsets(list(assignment))
                        lag_value = 0
                        for topic_partition in assignment:
                            current_position = await consumer.position(topic_partition)
                            lag_value += max(
                                end_offsets.get(topic_partition, 0) - current_position,
                                0,
                            )
                    else:
                        lag_value = 0
            except Exception:
                lag_value = -1

            self.kafka_consumer_lag.labels(group_id=group_id).set(lag_value)

    def prometheus_payload(self) -> bytes:
        return generate_latest(self.registry)

    @property
    def prometheus_content_type(self) -> str:
        return CONTENT_TYPE_LATEST
