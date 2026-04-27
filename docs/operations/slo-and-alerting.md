# SLO And Alerting

## Initial SLOs
- Availability: gateway, recommendation, and interaction ingest 5xx rate under 1% rolling 30 days.
- Latency: gateway, recommendation, and interaction ingest `p95 < 1000ms`, `p99 < 2500ms`.
- Freshness: worker heartbeat present for `content-worker`, `feature-worker`, and `model-trainer`.
- Durability: uploads must reach either local shared storage or S3-compatible storage before Kafka enqueue returns success.

## Alert Rules
Prometheus now loads `monitoring/prometheus-rules.yml` and sends alerts to Alertmanager.

Current alerts:
- `ServiceErrorBudgetBurn`
- `ServiceP95LatencyHigh`
- `ServiceP99LatencyHigh`
- `GatewayUnavailable`
- `RecommendationServiceUnavailable`
- `InteractionIngestServiceUnavailable`
- `WorkerHeartbeatMissing`
- `WorkerMetricsTargetDown`
- `WorkerProcessingFailures`
- `KafkaProducerDisconnected`
- `KafkaConsumerLagHigh`
- `KafkaDeadLetterTraffic`
- `RedisMemoryHigh`
- `PostgresExporterDown`
- `DatabaseErrors`
- `DatabaseP95LatencyHigh`

## Dashboards And Traces
- Grafana provisions Prometheus automatically from `monitoring/grafana/provisioning/datasources/prometheus.yml`.
- The default dashboard is `Video Commerce Overview` from `monitoring/grafana/dashboards/video-commerce-overview.json`.
- Jaeger is available at `http://127.0.0.1:16686` when the compose stack is running.
- Applications export OTLP traces to `otel-collector:4317`; the collector forwards traces to Jaeger.

## Validation
- Run `docker compose config -q` after editing compose or monitoring mounts.
- Run `pytest -q tests/test_observability_metrics.py tests/test_monitoring_config.py tests/test_telemetry_helpers.py`.
- If `promtool` is installed, run `promtool test rules tests/monitoring/prometheus-rules.test.yml`.

## Operator Response
- Service 5xx or latency alerts: inspect the service logs, `trace_id` in JSON logs, and service `/readyz`.
- Dependency down alerts: inspect `docker compose ps` and `docker compose logs <service>`.
- Worker heartbeat missing: inspect Redis, Kafka lag, and the worker logs; verify `video_commerce_worker_live_instances`.
- Kafka producer disconnected: verify broker health and topic availability before re-enabling traffic.
- Kafka lag or DLQ alerts: inspect `kafka-exporter`, worker processing errors, and the dead-letter topic.
- DB latency/error alerts: inspect Postgres exporter metrics and application `video_commerce_database_*` metrics.
