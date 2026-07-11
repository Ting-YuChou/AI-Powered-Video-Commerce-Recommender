from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prometheus_scrapes_services_workers_and_exporters():
    prometheus = (ROOT / "monitoring" / "prometheus.yml").read_text()

    for job_name in (
        "gateway-api",
        "recommendation-service",
        "interaction-ingest-service",
        "content-worker",
        "feature-worker",
        "model-trainer",
        "postgres",
        "redis-state",
        "redis-cache",
        "kafka",
        "flink",
        "catalog-event-publisher",
    ):
        assert f"job_name: {job_name}" in prometheus


def test_compose_provisions_tracing_and_grafana():
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "otel-collector:" in compose
    assert "jaeger:" in compose
    assert "postgres-exporter:" in compose
    assert "redis-cache-exporter:" in compose
    assert "kafka-exporter:" in compose
    assert "flink-volume-init:" in compose
    assert "flink-jobmanager:" in compose
    assert "flink-taskmanager:" in compose
    assert "flink-interaction-features:" in compose
    assert "flink-feature-history-materializer:" in compose
    assert "flink-ranking-pit-export:" in compose
    assert "pit-manifest-publisher:" in compose
    assert "iceberg-rest:" in compose
    assert "taskmanager.memory.jvm-metaspace.size: ${FLINK_TASKMANAGER_JVM_METASPACE_SIZE:-512m}" in compose
    assert compose.count("s3.endpoint: http://minio:9000") >= 3
    assert "ENVIRONMENT: ${FEATURE_LAKE_ENVIRONMENT:-development}" in compose
    assert "./monitoring/grafana/provisioning/datasources" in compose
    assert "./monitoring/grafana/dashboards" in compose


def test_flink_job_packages_parquet_format_for_pit_export():
    pom = (ROOT / "flink-jobs" / "interaction-features" / "pom.xml").read_text()

    assert "<artifactId>flink-parquet</artifactId>" in pom


def test_prometheus_rules_cover_service_worker_kafka_db_and_redis():
    rules = (ROOT / "monitoring" / "prometheus-rules.yml").read_text()

    for alert_name in (
        "ServiceErrorBudgetBurn",
        "ServiceP95LatencyHigh",
        "WorkerProcessingFailures",
        "KafkaConsumerLagHigh",
        "RedisMemoryHigh",
        "DatabaseErrors",
        "DatabaseP95LatencyHigh",
        "FeatureLakeMaterializationLagHigh",
        "CatalogOutboxBacklog",
        "FeatureLakeDlqTraffic",
        "PitManifestValidationFailure",
        "PitOnlineOfflineParityLow",
        "PitAssemblerParityLow",
        "PitLabelReconciliationLow",
        "PitCurrentStateDependencyDetected",
        "PitInvalidTrainingTensorDetected",
    ):
        assert f"alert: {alert_name}" in rules


def test_feature_lake_dashboard_and_metrics_are_declared():
    dashboard = ROOT / "monitoring/grafana/dashboards/feature-lake.json"
    assert dashboard.exists()
    observability = (ROOT / "video_commerce/common/observability.py").read_text()
    for metric in (
        "feature_lake_materialization_lag_seconds",
        "feature_lake_records_total",
        "feature_lake_dlq_total",
        "catalog_outbox_pending",
        "catalog_outbox_oldest_age_seconds",
        "pit_export_rows",
        "pit_manifest_validation_failures_total",
        "pit_online_offline_parity_ratio",
        "pit_leakage_rows",
        "pit_assembler_vector_parity_ratio",
        "pit_label_reconciliation_ratio",
        "pit_current_state_calls",
        "pit_invalid_feature_or_label_rows",
        "pit_value_mask_coverage_ratio",
    ):
        assert metric in observability
