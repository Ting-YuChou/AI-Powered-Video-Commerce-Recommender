import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_prometheus_scrapes_services_workers_and_exporters():
    prometheus = (ROOT / "monitoring" / "prometheus.yml").read_text()

    for job_name in (
        "gateway-api",
        "recommendation-service",
        "ranking-service",
        "ranking-triton",
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
        "pit-dataset-orchestrator",
        "pit-state-exporter",
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
    assert (
        "taskmanager.memory.jvm-metaspace.size: ${FLINK_TASKMANAGER_JVM_METASPACE_SIZE:-512m}"
        in compose
    )
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
        "FlinkInteractionFeatureJobMissing",
        "RecommendationOutboxBacklog",
        "CatalogOutboxBacklog",
        "FeatureLakeDlqTraffic",
        "PitManifestValidationFailure",
        "PitOnlineOfflineParityLow",
        "PitDailyRunMissing",
        "PitTrainerManifestFailure",
        "PitOrchestratorRunStuck",
        "PitOrchestratorLeaseExpired",
        "RankingArtifactFallbackPersisting",
        "RankingVerifiedModelMissing",
        "RankingHealthyRunnersLow",
        "RankingQueueWaitP95High",
        "RankingRunnerSlotP95High",
        "RankingOverloadSustained",
        "RankingCoordinatorPoolTimeout",
        "Ranking5xxDetected",
        "RankingTritonModelUnavailable",
        "RankingTritonOverloadSustained",
        "KafkaProducerDisconnected",
        "RateLimiterFailingOpen",
        "RankingRuntimeUnauthenticatedConnections",
        "RankingRuntimeAuthFailures",
        "RecommendationOutboxTerminal",
        "RecommendationImpressionAttributionDegraded",
        "CatalogOutboxTerminal",
        "ContentProcessingRedrivesExhausted",
        "ContentZeroNormEmbeddings",
        "PitRunWaitingForMaterializer",
        "KnownUserFeaturesMissing",
    ):
        assert f"alert: {alert_name}" in rules


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def _alert(name):
    rules = yaml.safe_load((ROOT / "monitoring" / "prometheus-rules.yml").read_text())
    for group in rules["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == name:
                return rule
    raise AssertionError(name)


def test_feature_worker_is_opt_in_and_its_alerts_are_gated_on_python_mode():
    services = _compose()["services"]

    assert services["feature-worker"]["profiles"] == ["python-features"]
    gate = 'video_commerce_feature_pipeline_mode{job="gateway-api",mode!="flink"}'
    for name in ("WorkerHeartbeatMissing", "WorkerMetricsTargetDown"):
        expr = _alert(name)["expr"]
        assert gate in expr, name
        # Every positive feature-worker matcher sits behind the gate.
        selects_feature_worker = re.findall(r'(?<!!)="feature-worker"', expr)
        assert len(selects_feature_worker) == expr.count(gate) == 1, name


def test_monitoring_images_are_pinned():
    services = _compose()["services"]

    for name in ("prometheus", "alertmanager", "grafana", "alert-webhook-sink"):
        image = services[name]["image"]
        assert ":" in image and not image.endswith(":latest"), image
    # url_file in webhook_config needs Alertmanager >= 0.26.
    major, minor = (
        int(part)
        for part in services["alertmanager"]["image"].split(":v")[1].split(".")[:2]
    )
    assert (major, minor) >= (0, 26)


def test_alertmanager_routes_critical_to_oncall_with_file_secret():
    config = yaml.safe_load((ROOT / "monitoring" / "alertmanager.yml").read_text())
    routes = {
        tuple(route["matchers"]): route["receiver"]
        for route in config["route"]["routes"]
    }
    receivers = {receiver["name"]: receiver for receiver in config["receivers"]}

    assert config["route"]["receiver"] == "default"
    assert routes[('severity="critical"',)] == "oncall-webhook"
    assert routes[('severity="warning"',)] == "default"
    webhook = receivers["oncall-webhook"]["webhook_configs"][0]
    assert "url" not in webhook
    assert webhook["url_file"] == "/etc/alertmanager/secrets/oncall-webhook-url"
    inhibits = config["inhibit_rules"]
    assert {
        "source_matchers": ['severity="critical"'],
        "target_matchers": ['severity="warning"'],
        "equal": ["alertname"],
    } in inhibits
    flink = next(
        rule
        for rule in inhibits
        if rule["source_matchers"] == ['alertname="FlinkInteractionFeatureJobMissing"']
    )
    assert "FlinkCheckpointUnavailable" in flink["target_matchers"][0]

    volumes = _compose()["services"]["alertmanager"]["volumes"]
    assert any(
        "oncall-webhook-url.example" in volume
        and volume.endswith(":/etc/alertmanager/secrets/oncall-webhook-url:ro")
        for volume in volumes
    )
    example = ROOT / "monitoring" / "secrets" / "oncall-webhook-url.example"
    assert example.read_text().strip().startswith("http")


def test_multiprocess_dir_is_scoped_to_uvicorn_services():
    compose = _compose()
    services = compose["services"]
    multiproc = {
        "gateway-api",
        "recommendation-service",
        "ranking-service",
        "interaction-ingest-service",
    }

    for name, service in services.items():
        environment = service.get("environment") or {}
        if name in multiproc:
            assert environment["PROMETHEUS_MULTIPROC_DIR"] == "/run/prometheus-multiproc"
            assert any(
                entry.startswith("/run/prometheus-multiproc:")
                for entry in service["tmpfs"]
            ), name
        else:
            assert "PROMETHEUS_MULTIPROC_DIR" not in environment, name


def test_helm_mounts_memory_multiproc_dir_and_relabels_job():
    values = yaml.safe_load(
        (ROOT / "charts/video-commerce/values.yaml").read_text(encoding="utf-8")
    )
    workloads = values["backend"]["workloads"]
    flagged = {
        workload["name"]
        for workload in workloads.values()
        if workload.get("prometheusMultiproc")
    }

    assert flagged == {
        "gateway-api",
        "recommendation-service",
        "ranking-service",
        "interaction-ingest-service",
    }
    assert values["backend"]["emptyDirs"]["prometheusMultiproc"]["medium"] == "Memory"
    service_monitor = (
        ROOT / "charts/video-commerce/templates/servicemonitor.yaml"
    ).read_text(encoding="utf-8")
    assert "targetLabel: job" in service_monitor
    assert "replacement: {{ $workload.name }}" in service_monitor


def test_feature_lake_dashboard_and_metrics_are_declared():
    dashboard = ROOT / "monitoring/grafana/dashboards/feature-lake.json"
    assert dashboard.exists()
    observability = (ROOT / "video_commerce/common/observability.py").read_text()
    dashboard_payload = dashboard.read_text()
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
        "pit_orchestrator_runs_total",
        "pit_orchestrator_last_success_timestamp",
        "pit_orchestrator_waiting_for_rows",
        "pit_trainer_waiting_for_manifest",
        "pit_training_duplicate_manifest_skips_total",
        "ranking_untrained_fallback_total",
        "pit_orchestrator_run_in_progress",
        "pit_orchestrator_lease_expired",
    ):
        assert metric in observability
    for metric in (
        "pit_orchestrator_waiting_for_rows",
        "pit_trainer_waiting_for_manifest",
        "pit_training_duplicate_manifest_skips_total",
        "ranking_untrained_fallback_total",
    ):
        assert metric in dashboard_payload
