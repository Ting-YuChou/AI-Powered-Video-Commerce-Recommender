# SLO And Alerting

## Initial SLOs
- Availability: gateway, recommendation, and interaction ingest 5xx rate under 1% rolling 30 days.
- Latency: gateway, recommendation, and interaction ingest `p95 < 1000ms`, `p99 < 2500ms`.
- Freshness (online features): with the default Flink pipeline
  (`FEATURE_PIPELINE_MODE=flink`), exactly one
  `video-commerce-interaction-features` job is RUNNING
  (`FlinkInteractionFeatureJobMissing`) and its last completed checkpoint is
  under 5 minutes old (`FlinkCheckpointUnavailable`). Only in Python feature
  mode does the `feature-worker` heartbeat count instead.
- Freshness (workers): `content-worker` and `model-trainer` heartbeats are
  present (`WorkerHeartbeatMissing`).
- Durability: uploads must reach either local shared storage or S3-compatible storage before Kafka enqueue returns success.

## Alert Rules
Prometheus loads `monitoring/prometheus-rules.yml` and sends alerts to
Alertmanager. Every alert has a `severity` label (`critical` or `warning`) and a
`runbook_url` annotation pointing at its section in
[Alert Runbooks](alert-runbooks.md), which lists every alert with its symptom,
impact, first checks, and remediation.

Kubernetes: `charts/video-commerce/files/prometheus-rules.yml` is a byte-for-byte
copy of the rules (`tests/test_alert_runbooks.py` enforces parity). Set
`monitoring.prometheusRule.enabled=true` to install it as a prometheus-operator
`PrometheusRule`. ServiceMonitors relabel `job` to the workload name
(`gateway-api`, `recommendation-service`, ...), so the same `job=` matchers work
in Compose and Kubernetes.

The `feature-worker` targets in `WorkerHeartbeatMissing` and
`WorkerMetricsTargetDown` only alert while the gateway reports
`video_commerce_feature_pipeline_mode{mode!="flink"} == 1`. In Compose the worker
is in the `python-features` profile and is not started by default.

## Alert Routing And Receivers
`monitoring/alertmanager.yml`:

- `severity="critical"` routes to `oncall-webhook` (group wait 10s, repeat 1h).
- `severity="warning"` routes to `default`, which has no notifier: warnings are
  visible in the Alertmanager UI/API and Grafana but do not page.
- Inhibition: a critical alert silences the warning of the same `alertname`;
  `FlinkInteractionFeatureJobMissing` silences `FlinkCheckpointUnavailable` and
  `FlinkInteractionWatermarkLag`.

The on-call webhook URL is a secret, so it is read from a file with
`url_file: /etc/alertmanager/secrets/oncall-webhook-url` (needs Alertmanager
0.26 or newer; Compose pins `prom/alertmanager:v0.27.0`, and 0.25 rejects the
field). The file contains only the URL.

### Compose
By default Compose mounts `monitoring/secrets/oncall-webhook-url.example`, which
points at the local sink `http://alert-webhook-sink:9095/oncall`. Start the sink
to see exactly what on-call would receive:

```bash
docker compose --profile alert-sink up -d alert-webhook-sink
docker compose logs -f alert-webhook-sink
```

Without the sink, critical notifications fail to deliver and Alertmanager logs
the error; nothing else changes.

For a real receiver, keep the URL outside the repository and point Compose at
it:

```bash
install -m 0400 /dev/null /etc/video-commerce/oncall-webhook-url
printf '%s\n' 'https://hooks.example.com/...' > /etc/video-commerce/oncall-webhook-url
ALERTMANAGER_ONCALL_WEBHOOK_URL_FILE=/etc/video-commerce/oncall-webhook-url \
  docker compose up -d alertmanager
```

### Kubernetes
The chart does not install Alertmanager. With prometheus-operator, create the
secret and list it in the `Alertmanager` resource's `spec.secrets` (for
kube-prometheus-stack: `alertmanager.alertmanagerSpec.secrets`); the operator
mounts it at `/etc/alertmanager/secrets/<secret>/<key>`:

```bash
kubectl -n monitoring create secret generic oncall-webhook \
  --from-literal=url='https://hooks.example.com/...'
```

Then use `url_file: /etc/alertmanager/secrets/oncall-webhook/url` in the
receiver. Rotate by updating the secret; Alertmanager re-reads the file on each
notification.

### Delivery drill
Send a synthetic critical alert and confirm the sink logs a payload whose
annotations include `runbook_url`:

```bash
docker compose exec alertmanager amtool alert add DeliveryDrill \
  severity=critical --annotation=runbook_url=https://example.invalid/drill \
  --alertmanager.url=http://127.0.0.1:9093
docker compose logs --since 2m alert-webhook-sink
```

## Dashboards And Traces
- Grafana provisions Prometheus automatically from `monitoring/grafana/provisioning/datasources/prometheus.yml`.
- The default dashboard is `Video Commerce Overview` from `monitoring/grafana/dashboards/video-commerce-overview.json`.
- Jaeger is available at `http://127.0.0.1:16686` when the compose stack is running.
- Applications export OTLP traces to `otel-collector:4317`; the collector forwards traces to Jaeger.
- Prometheus (`v2.54.1`), Alertmanager (`v0.27.0`), and Grafana (`11.2.0`) images are pinned in Compose; upgrade them deliberately and rerun the validation below.

## Validation
- Run `docker compose config -q` after editing compose or monitoring mounts.
- Run `pytest -q tests/test_observability_metrics.py tests/test_prometheus_multiprocess.py tests/test_monitoring_config.py tests/test_alert_runbooks.py tests/test_telemetry_helpers.py`.
- Run `promtool test rules tests/monitoring/prometheus-rules.test.yml`; every new or changed alert needs a case there.
- Run `amtool check-config` as shown in [Observability Validation](observability-validation.md#prometheus).

## Operator Response
Start from the alert's `runbook_url` ([Alert Runbooks](alert-runbooks.md)).
Multi-worker metric caveats are in
[Observability Validation](observability-validation.md#multi-worker-metrics).
