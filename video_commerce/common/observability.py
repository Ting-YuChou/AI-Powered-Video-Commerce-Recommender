"""
Observability utilities for request tracing, structured logging, and metrics.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
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
        trace_context = get_current_trace_context()
        if trace_context:
            record.trace_id = trace_context["trace_id"]
            record.span_id = trace_context["span_id"]
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
        if getattr(record, "trace_id", None):
            payload["trace_id"] = record.trace_id
        if getattr(record, "span_id", None):
            payload["span_id"] = record.span_id

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


def get_current_trace_context() -> Optional[Dict[str, str]]:
    """Return active OpenTelemetry trace identifiers when tracing is enabled."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import INVALID_SPAN_CONTEXT
    except Exception:
        return None

    span = trace.get_current_span()
    if not span:
        return None
    context = span.get_span_context()
    if not context or context == INVALID_SPAN_CONTEXT or not context.is_valid:
        return None
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }


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

    for logger_name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(log_level)
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = True
    access_logger.setLevel(logging.WARNING)

    third_party_level = logging.WARNING if log_level <= logging.INFO else log_level
    for logger_name in ("httpcore", "httpx", "aiokafka", "opentelemetry"):
        logging.getLogger(logger_name).setLevel(third_party_level)


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
            ["role"],
            registry=self.registry,
        )
        self.redis_connected_clients = Gauge(
            "video_commerce_redis_connected_clients",
            "Redis connected clients",
            ["role"],
            registry=self.registry,
        )
        self.redis_used_memory_bytes = Gauge(
            "video_commerce_redis_used_memory_bytes",
            "Redis used memory in bytes",
            ["role"],
            registry=self.registry,
        )
        self.redis_maxmemory_bytes = Gauge(
            "video_commerce_redis_maxmemory_bytes",
            "Redis configured max memory in bytes",
            ["role"],
            registry=self.registry,
        )
        self.redis_memory_fragmentation_ratio = Gauge(
            "video_commerce_redis_memory_fragmentation_ratio",
            "Redis memory fragmentation ratio",
            ["role"],
            registry=self.registry,
        )
        self.kafka_producer_connected = Gauge(
            "video_commerce_kafka_producer_connected",
            "Kafka producer connection status",
            registry=self.registry,
        )
        self.kafka_messages_produced_total = Counter(
            "video_commerce_kafka_messages_produced_total",
            "Kafka messages produced by topic and status",
            ["topic", "status"],
            registry=self.registry,
        )
        self.kafka_messages_consumed_total = Counter(
            "video_commerce_kafka_messages_consumed_total",
            "Kafka messages consumed by topic, group, and status",
            ["topic", "group_id", "status"],
            registry=self.registry,
        )
        self.kafka_message_processing_seconds = Histogram(
            "video_commerce_kafka_message_processing_seconds",
            "Kafka message processing latency in seconds",
            ["topic", "group_id"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.kafka_message_retries_total = Counter(
            "video_commerce_kafka_message_retries_total",
            "Kafka message handler retry attempts",
            ["topic", "group_id"],
            registry=self.registry,
        )
        self.kafka_dead_letter_messages_total = Counter(
            "video_commerce_kafka_dead_letter_messages_total",
            "Kafka messages published to the dead-letter topic",
            ["source_topic", "dead_letter_topic"],
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
        self.worker_heartbeat_timestamp_seconds = Gauge(
            "video_commerce_worker_heartbeat_timestamp_seconds",
            "Unix timestamp of the latest worker heartbeat observed by the worker itself",
            ["service", "instance_id"],
            registry=self.registry,
        )
        self.worker_messages_processed_total = Counter(
            "video_commerce_worker_messages_processed_total",
            "Messages processed by background workers",
            ["service", "topic", "status"],
            registry=self.registry,
        )
        self.worker_message_processing_seconds = Histogram(
            "video_commerce_worker_message_processing_seconds",
            "Worker message processing latency in seconds",
            ["service", "topic"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.asr_transcriptions_total = Counter(
            "video_commerce_asr_transcriptions_total",
            "Uploaded content ASR attempts by outcome",
            ["status"],
            registry=self.registry,
        )
        self.asr_transcription_duration_seconds = Histogram(
            "video_commerce_asr_transcription_duration_seconds",
            "Audio extraction and ASR latency in seconds",
            ["status"],
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.worker_training_runs_total = Counter(
            "video_commerce_worker_training_runs_total",
            "Model trainer runs by trigger and status",
            ["trigger", "status"],
            registry=self.registry,
        )
        self.worker_training_duration_seconds = Histogram(
            "video_commerce_worker_training_duration_seconds",
            "Model trainer run duration in seconds",
            ["trigger"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600),
            registry=self.registry,
        )
        self.database_query_duration_seconds = Histogram(
            "video_commerce_database_query_duration_seconds",
            "Postgres operation latency in seconds",
            ["operation", "status"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.database_queries_total = Counter(
            "video_commerce_database_queries_total",
            "Postgres operations by operation and status",
            ["operation", "status"],
            registry=self.registry,
        )
        self.database_pool_connections = Gauge(
            "video_commerce_database_pool_connections",
            "SQLAlchemy pool connection counts",
            ["state"],
            registry=self.registry,
        )
        self.recommendation_requests_total = Counter(
            "video_commerce_recommendation_requests_total",
            "Recommendation requests by outcome, cache state, and serving path",
            ["result", "cache_hit", "serving_path"],
            registry=self.registry,
        )
        self.recommendation_candidates = Histogram(
            "video_commerce_recommendation_candidates",
            "Candidate counts seen by recommendation serving",
            ["stage"],
            buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry,
        )
        self.ranking_batch_queue_depth = Gauge(
            "video_commerce_ranking_batch_queue_depth",
            "Queued ranking requests in the local micro-batcher",
            registry=self.registry,
        )
        self.ranking_batch_size = Histogram(
            "video_commerce_ranking_batch_size",
            "Ranking requests combined per micro-batch",
            ["path"],
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=self.registry,
        )
        self.ranking_batch_candidates = Histogram(
            "video_commerce_ranking_batch_candidates",
            "Candidate products processed per ranking batch",
            ["path"],
            buckets=(1, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
            registry=self.registry,
        )
        self.ranking_batch_queue_wait_seconds = Histogram(
            "video_commerce_ranking_batch_queue_wait_seconds",
            "Maximum request queue wait before a ranking batch dispatches",
            ["path"],
            buckets=(
                0.0005,
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
            ),
            registry=self.registry,
        )
        self.ranking_batch_fill_ratio = Histogram(
            "video_commerce_ranking_batch_fill_ratio",
            "Fraction of configured maximum batch size used by each ranking batch",
            ["path"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
            registry=self.registry,
        )
        self.ranking_batch_target_fill_ratio = Histogram(
            "video_commerce_ranking_batch_target_fill_ratio",
            "Fraction of configured target batch size used by each ranking batch",
            ["path"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0),
            registry=self.registry,
        )
        self.ranking_batch_stage_seconds = Histogram(
            "video_commerce_ranking_batch_stage_seconds",
            "Ranking micro-batch execution time by stage",
            ["path", "stage"],
            buckets=(
                0.0005,
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2,
                5,
            ),
            registry=self.registry,
        )
        self.ranking_active_batches = Gauge(
            "video_commerce_ranking_active_batches",
            "Ranking batches currently executing",
            registry=self.registry,
        )
        self.ranking_cancelled_total = Counter(
            "video_commerce_ranking_cancelled_total",
            "Ranking requests cancelled or expired before inference",
            ["reason"],
            registry=self.registry,
        )
        self.ranking_direct_total = Counter(
            "video_commerce_ranking_direct_total",
            "Ranking requests handled outside normal micro-batch execution",
            ["reason"],
            registry=self.registry,
        )
        self.ranking_runner_queue_depth = Gauge(
            "video_commerce_ranking_runner_queue_depth",
            "Queued ranking micro-batches inside each ranking runner",
            registry=self.registry,
        )
        self.ranking_runner_batches_total = Counter(
            "video_commerce_ranking_runner_batches_total",
            "Ranking runner micro-batches by outcome",
            ["status"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_inflight = Gauge(
            "video_commerce_ranking_runner_endpoint_inflight_batches",
            "Coordinator in-flight ranking batches by runner endpoint",
            ["endpoint"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_available_connections = Gauge(
            "video_commerce_ranking_runner_endpoint_available_connections",
            "Coordinator available ranking-runner persistent connections by endpoint",
            ["endpoint"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_events_total = Counter(
            "video_commerce_ranking_runner_endpoint_events_total",
            "Coordinator runner endpoint events",
            ["endpoint", "event"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_state = Gauge(
            "video_commerce_ranking_runner_endpoint_state",
            "Coordinator runner endpoint state as a one-hot gauge",
            ["endpoint", "state"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_missing_refreshes = Gauge(
            "video_commerce_ranking_runner_endpoint_missing_refreshes",
            "Consecutive DNS refreshes missing a known runner endpoint",
            ["endpoint"],
            registry=self.registry,
        )
        self.ranking_runner_endpoint_removed_total = Counter(
            "video_commerce_ranking_runner_endpoint_removed_total",
            "Coordinator runner endpoint removals by reason",
            ["endpoint", "reason"],
            registry=self.registry,
        )
        self.ranking_runner_late_writes_total = Counter(
            "video_commerce_ranking_runner_late_writes_total",
            "Ranking runner responses that could not be written because the client disconnected",
            ["exception_type"],
            registry=self.registry,
        )
        self.ranking_wait_for_runner_slot_seconds = Histogram(
            "video_commerce_ranking_wait_for_runner_slot_seconds",
            "Time spent waiting for a ranking-runner dispatch slot",
            buckets=(
                0.0005,
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2,
                5,
            ),
            registry=self.registry,
        )
        self.ranking_runner_payload_bytes = Histogram(
            "video_commerce_ranking_runner_payload_bytes",
            "Encoded ranking-runner request payload bytes by internal payload version",
            ["payload_version"],
            buckets=(
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                33554432,
            ),
            registry=self.registry,
        )
        self.ranking_coordinator_client_errors_total = Counter(
            "video_commerce_ranking_coordinator_client_errors_total",
            "Ranking-service coordinator client errors by reason",
            ["reason"],
            registry=self.registry,
        )
        self.upstream_requests_total = Counter(
            "video_commerce_upstream_requests_total",
            "Internal upstream HTTP requests by target and status",
            ["target", "status"],
            registry=self.registry,
        )
        self.upstream_inflight = Gauge(
            "video_commerce_upstream_inflight",
            "Internal upstream HTTP requests currently in flight",
            ["target"],
            registry=self.registry,
        )
        self.best_effort_tasks_total = Counter(
            "video_commerce_best_effort_tasks_total",
            "Best-effort background tasks by task name and outcome",
            ["task", "status"],
            registry=self.registry,
        )
        self.best_effort_queue_depth = Gauge(
            "video_commerce_best_effort_queue_depth",
            "Queued best-effort background tasks in this worker",
            registry=self.registry,
        )
        self.interactions_ingested_total = Counter(
            "video_commerce_interactions_ingested_total",
            "Interaction ingest events by action and status",
            ["action", "status"],
            registry=self.registry,
        )
        self.content_uploads_total = Counter(
            "video_commerce_content_uploads_total",
            "Content upload enqueue outcomes by priority and status",
            ["priority", "status"],
            registry=self.registry,
        )
        self.feature_lake_materialization_lag_seconds = Gauge(
            "feature_lake_materialization_lag_seconds",
            "Latest observed feature-lake materialization lag",
            ["record_type"],
            registry=self.registry,
        )
        self.feature_lake_records_total = Counter(
            "feature_lake_records_total",
            "Feature-lake records by type and status",
            ["record_type", "status"],
            registry=self.registry,
        )
        self.feature_lake_dlq_total = Counter(
            "feature_lake_dlq_total",
            "Feature-lake records routed to DLQ",
            ["source_topic"],
            registry=self.registry,
        )
        self.catalog_outbox_pending = Gauge(
            "catalog_outbox_pending",
            "Unpublished catalog outbox rows from completed activations",
            registry=self.registry,
        )
        self.catalog_outbox_oldest_age_seconds = Gauge(
            "catalog_outbox_oldest_age_seconds",
            "Age of the oldest unpublished catalog outbox row",
            registry=self.registry,
        )
        self.pit_export_rows = Gauge(
            "pit_export_rows",
            "Rows in the latest PIT Parquet export",
            registry=self.registry,
        )
        self.pit_manifest_validation_failures_total = Counter(
            "pit_manifest_validation_failures_total",
            "Fail-closed PIT manifest or shard validation failures",
            ["reason"],
            registry=self.registry,
        )
        self.pit_online_offline_parity_ratio = Gauge(
            "pit_online_offline_parity_ratio",
            "Finalized online/offline feature bundle parity ratio",
            registry=self.registry,
        )
        self.pit_leakage_rows = Gauge(
            "pit_leakage_rows",
            "Rows violating PIT event-time or availability constraints",
            registry=self.registry,
        )
        self.pit_assembler_vector_parity_ratio = Gauge(
            "pit_assembler_vector_parity_ratio",
            "Exact shared-assembler vector parity for identical bundles",
            registry=self.registry,
        )
        self.pit_label_reconciliation_ratio = Gauge(
            "pit_label_reconciliation_ratio",
            "Typed labels reconciled with finalized attribution facts",
            registry=self.registry,
        )
        self.pit_current_state_calls = Gauge(
            "pit_current_state_calls",
            "Forbidden current Redis/catalog calls observed during PIT training",
            registry=self.registry,
        )
        self.pit_invalid_feature_or_label_rows = Gauge(
            "pit_invalid_feature_or_label_rows",
            "PIT rows containing invalid feature or label tensors",
            registry=self.registry,
        )
        self.pit_value_mask_coverage_ratio = Gauge(
            "pit_value_mask_coverage_ratio",
            "Purchase rows containing an actual attributed business value",
            registry=self.registry,
        )
        self._process = psutil.Process()

    def record_request(
        self, method: str, path: str, status_code: int, duration: float
    ) -> None:
        status = str(status_code)
        self.http_requests_total.labels(method=method, path=path, status=status).inc()
        self.http_request_duration_seconds.labels(method=method, path=path).observe(
            duration
        )

    def record_exception(self, method: str, path: str, exception_type: str) -> None:
        self.http_request_exceptions_total.labels(
            method=method,
            path=path,
            exception_type=exception_type,
        ).inc()

    def record_database_query(
        self, operation: str, duration: float, status: str = "success"
    ) -> None:
        self.database_queries_total.labels(operation=operation, status=status).inc()
        self.database_query_duration_seconds.labels(
            operation=operation,
            status=status,
        ).observe(duration)

    def update_database_pool_metrics(self, system_store) -> None:
        engine = getattr(system_store, "engine", None)
        if engine is None:
            return
        pool = getattr(engine.sync_engine, "pool", None)
        if pool is None:
            return
        for state, getter in {
            "checked_out": getattr(pool, "checkedout", None),
            "checked_in": getattr(pool, "checkedin", None),
            "overflow": getattr(pool, "overflow", None),
            "size": getattr(pool, "size", None),
        }.items():
            if callable(getter):
                self.database_pool_connections.labels(state=state).set(getter())

    def record_kafka_produce(self, topic: str, status: str) -> None:
        self.kafka_messages_produced_total.labels(topic=topic, status=status).inc()

    def record_kafka_consume(
        self, topic: str, group_id: str, status: str, duration: float
    ) -> None:
        self.kafka_messages_consumed_total.labels(
            topic=topic,
            group_id=group_id,
            status=status,
        ).inc()
        self.kafka_message_processing_seconds.labels(
            topic=topic,
            group_id=group_id,
        ).observe(duration)

    def record_kafka_retry(self, topic: str, group_id: str) -> None:
        self.kafka_message_retries_total.labels(topic=topic, group_id=group_id).inc()

    def record_kafka_dead_letter(
        self, source_topic: str, dead_letter_topic: str
    ) -> None:
        self.kafka_dead_letter_messages_total.labels(
            source_topic=source_topic,
            dead_letter_topic=dead_letter_topic,
        ).inc()

    def record_worker_message(
        self, service: str, topic: str, status: str, duration: float
    ) -> None:
        self.worker_messages_processed_total.labels(
            service=service,
            topic=topic,
            status=status,
        ).inc()
        self.worker_message_processing_seconds.labels(
            service=service, topic=topic
        ).observe(duration)

    def record_asr_transcription(self, status: str, duration: float = 0.0) -> None:
        self.asr_transcriptions_total.labels(status=status).inc()
        self.asr_transcription_duration_seconds.labels(status=status).observe(duration)

    def update_worker_heartbeat(self, service: str, instance_id: str) -> None:
        self.worker_heartbeat_timestamp_seconds.labels(
            service=service,
            instance_id=instance_id,
        ).set(time.time())

    def record_training_run(self, trigger: str, status: str, duration: float) -> None:
        self.worker_training_runs_total.labels(trigger=trigger, status=status).inc()
        self.worker_training_duration_seconds.labels(trigger=trigger).observe(duration)

    def update_catalog_outbox(self, *, pending: int, oldest_age_seconds: float) -> None:
        self.catalog_outbox_pending.set(max(0, int(pending)))
        self.catalog_outbox_oldest_age_seconds.set(max(0.0, float(oldest_age_seconds)))

    def record_pit_manifest_validation_failure(self, reason: str) -> None:
        self.pit_manifest_validation_failures_total.labels(reason=reason).inc()

    def update_typed_pit_training_metrics(
        self,
        *,
        assembler_parity_ratio: float,
        label_reconciliation_ratio: float,
        current_state_calls: int,
        invalid_rows: int,
        value_mask_coverage: float,
    ) -> None:
        self.pit_assembler_vector_parity_ratio.set(
            max(0.0, min(1.0, float(assembler_parity_ratio)))
        )
        self.pit_label_reconciliation_ratio.set(
            max(0.0, min(1.0, float(label_reconciliation_ratio)))
        )
        self.pit_current_state_calls.set(max(0, int(current_state_calls)))
        self.pit_invalid_feature_or_label_rows.set(max(0, int(invalid_rows)))
        self.pit_value_mask_coverage_ratio.set(
            max(0.0, min(1.0, float(value_mask_coverage)))
        )

    def record_recommendation(
        self,
        *,
        result: str,
        cache_hit: bool,
        serving_path: str,
        candidate_count: int = 0,
        ranked_count: int = 0,
    ) -> None:
        self.recommendation_requests_total.labels(
            result=result,
            cache_hit=str(bool(cache_hit)).lower(),
            serving_path=serving_path,
        ).inc()
        self.recommendation_candidates.labels(stage="retrieved").observe(
            candidate_count
        )
        self.recommendation_candidates.labels(stage="ranked").observe(ranked_count)

    def set_ranking_queue_depth(self, depth: int) -> None:
        self.ranking_batch_queue_depth.set(max(0, int(depth)))

    def record_ranking_batch(
        self,
        *,
        request_count: int,
        candidate_count: int,
        queue_wait_seconds: float,
        path: str,
        max_batch_requests: int | None = None,
        target_batch_requests: int | None = None,
    ) -> None:
        self.ranking_batch_size.labels(path=path).observe(max(0, int(request_count)))
        self.ranking_batch_candidates.labels(path=path).observe(
            max(0, int(candidate_count))
        )
        self.ranking_batch_queue_wait_seconds.labels(path=path).observe(
            max(0.0, float(queue_wait_seconds))
        )
        if max_batch_requests:
            fill_ratio = max(0.0, min(1.0, float(request_count) / max_batch_requests))
            self.ranking_batch_fill_ratio.labels(path=path).observe(fill_ratio)
        if target_batch_requests:
            target_fill_ratio = max(
                0.0, min(2.0, float(request_count) / target_batch_requests)
            )
            self.ranking_batch_target_fill_ratio.labels(path=path).observe(
                target_fill_ratio
            )

    def record_ranking_batch_stage(
        self,
        *,
        path: str,
        stage: str,
        duration_seconds: float,
    ) -> None:
        self.ranking_batch_stage_seconds.labels(path=path, stage=stage).observe(
            max(0.0, float(duration_seconds))
        )

    def inc_ranking_active_batches(self) -> None:
        self.ranking_active_batches.inc()

    def dec_ranking_active_batches(self) -> None:
        self.ranking_active_batches.dec()

    def record_ranking_cancelled(self, reason: str) -> None:
        self.ranking_cancelled_total.labels(reason=reason).inc()

    def record_ranking_direct(self, reason: str) -> None:
        self.ranking_direct_total.labels(reason=reason).inc()

    def set_ranking_runner_queue_depth(self, depth: int) -> None:
        self.ranking_runner_queue_depth.set(max(0, int(depth)))

    def record_ranking_runner_batch(self, status: str) -> None:
        self.ranking_runner_batches_total.labels(status=status).inc()

    def set_ranking_runner_endpoint_inflight(
        self, endpoint: str, inflight_batches: int
    ) -> None:
        self.ranking_runner_endpoint_inflight.labels(endpoint=endpoint).set(
            max(0, int(inflight_batches))
        )

    def set_ranking_runner_endpoint_available_connections(
        self, endpoint: str, available_connections: int
    ) -> None:
        self.ranking_runner_endpoint_available_connections.labels(
            endpoint=endpoint
        ).set(max(0, int(available_connections)))

    def record_ranking_runner_endpoint_event(self, endpoint: str, event: str) -> None:
        self.ranking_runner_endpoint_events_total.labels(
            endpoint=endpoint, event=event
        ).inc()

    def set_ranking_runner_endpoint_state(self, endpoint: str, state: str) -> None:
        states = ("active", "draining", "failed", "overloaded")
        for candidate in states:
            self.ranking_runner_endpoint_state.labels(
                endpoint=endpoint, state=candidate
            ).set(1 if candidate == state else 0)

    def set_ranking_runner_endpoint_missing_refreshes(
        self, endpoint: str, missing_refreshes: int
    ) -> None:
        self.ranking_runner_endpoint_missing_refreshes.labels(endpoint=endpoint).set(
            max(0, int(missing_refreshes))
        )

    def record_ranking_runner_endpoint_removed(
        self, endpoint: str, reason: str
    ) -> None:
        self.ranking_runner_endpoint_removed_total.labels(
            endpoint=endpoint, reason=reason
        ).inc()

    def record_ranking_runner_late_write(self, exception_type: str) -> None:
        self.ranking_runner_late_writes_total.labels(
            exception_type=exception_type
        ).inc()

    def record_ranking_wait_for_runner_slot(self, duration_seconds: float) -> None:
        self.ranking_wait_for_runner_slot_seconds.observe(
            max(0.0, float(duration_seconds))
        )

    def observe_ranking_runner_payload_bytes(
        self,
        *,
        payload_version: str,
        size_bytes: int,
    ) -> None:
        self.ranking_runner_payload_bytes.labels(
            payload_version=str(payload_version)
        ).observe(max(0, int(size_bytes)))

    def record_ranking_coordinator_client_error(self, reason: str) -> None:
        self.ranking_coordinator_client_errors_total.labels(reason=reason).inc()

    def inc_upstream_inflight(self, target: str) -> None:
        self.upstream_inflight.labels(target=target).inc()

    def dec_upstream_inflight(self, target: str) -> None:
        self.upstream_inflight.labels(target=target).dec()

    def record_upstream_request(self, target: str, status: str | int) -> None:
        self.upstream_requests_total.labels(target=target, status=str(status)).inc()

    def record_best_effort_task(self, task: str, status: str) -> None:
        self.best_effort_tasks_total.labels(task=task, status=status).inc()

    def set_best_effort_queue_depth(self, depth: int) -> None:
        self.best_effort_queue_depth.set(max(0, int(depth)))

    def record_interaction_ingest(self, action: str, status: str) -> None:
        self.interactions_ingested_total.labels(action=action, status=status).inc()

    def record_content_upload(self, priority: str, status: str) -> None:
        self.content_uploads_total.labels(priority=priority, status=status).inc()

    async def collect_runtime_metrics(
        self,
        feature_store=None,
        kafka_manager=None,
        system_store=None,
        worker_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Refresh gauges from the current process and backing services."""
        self.process_cpu_percent.set(self._process.cpu_percent())
        self.process_resident_memory_bytes.set(self._process.memory_info().rss)

        if feature_store and getattr(feature_store, "redis_client", None):
            await self._collect_redis_metrics("state", feature_store.redis_client)
            cache_client = getattr(feature_store, "cache_redis_client", None)
            if (
                cache_client is not None
                and cache_client is not feature_store.redis_client
            ):
                await self._collect_redis_metrics("cache", cache_client)

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

        if system_store:
            self.update_database_pool_metrics(system_store)

    async def _collect_redis_metrics(self, role: str, redis_client) -> None:
        try:
            redis_info = await redis_client.info()
            self.redis_ops_per_sec.labels(role=role).set(
                redis_info.get("instantaneous_ops_per_sec", 0)
            )
            self.redis_connected_clients.labels(role=role).set(
                redis_info.get("connected_clients", 0)
            )
            self.redis_used_memory_bytes.labels(role=role).set(
                redis_info.get("used_memory", 0)
            )
            self.redis_maxmemory_bytes.labels(role=role).set(
                redis_info.get("maxmemory", 0)
            )
            self.redis_memory_fragmentation_ratio.labels(role=role).set(
                redis_info.get("mem_fragmentation_ratio", 0)
            )
        except Exception:
            pass

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

    def start_metrics_server(self, port: int, addr: str = "0.0.0.0") -> None:
        start_http_server(port, addr=addr, registry=self.registry)


def start_worker_metrics_server(
    observability: ObservabilityManager,
    monitoring_config,
    *,
    default_port: int,
) -> Optional[int]:
    """Start a standalone Prometheus endpoint for a non-HTTP worker."""
    if not getattr(monitoring_config, "enable_prometheus_metrics", True):
        return None
    configured_port = int(os.getenv("MONITORING_WORKER_METRICS_PORT", default_port))
    if configured_port <= 0:
        return None
    host = os.getenv(
        "MONITORING_WORKER_METRICS_HOST",
        getattr(monitoring_config, "worker_metrics_host", "0.0.0.0"),
    )
    observability.start_metrics_server(configured_port, addr=host)
    return configured_port
