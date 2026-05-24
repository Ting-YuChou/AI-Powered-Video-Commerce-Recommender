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
    assert "./monitoring/grafana/provisioning/datasources" in compose
    assert "./monitoring/grafana/dashboards" in compose


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
    ):
        assert f"alert: {alert_name}" in rules
