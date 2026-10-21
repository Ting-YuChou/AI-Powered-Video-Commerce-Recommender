# Observability Validation

Use this checklist after changing metrics, alerts, or tracing.

## Local Stack

```bash
docker compose config -q
docker compose up -d prometheus alertmanager grafana jaeger otel-collector
docker compose up -d gateway-api recommendation-service interaction-ingest-service content-worker model-trainer
# Only for the Python feature pipeline (Flink is the default):
FEATURE_PIPELINE_MODE=python docker compose --profile python-features up -d feature-worker
```

## Prometheus

Open `http://127.0.0.1:9090/targets` and confirm these jobs are up:

- `gateway-api`, `recommendation-service`, `interaction-ingest-service`
- `content-worker`, `model-trainer` (and `feature-worker` in Python feature
  mode; in Flink mode its target is down by design and its alerts are gated)
- `postgres`, `redis-state`, `redis-cache`, `kafka`

Run representative queries:

```promql
sum by (job) (rate(video_commerce_http_requests_total[5m]))
sum by (job, service, topic, status) (rate(video_commerce_worker_messages_processed_total[5m]))
histogram_quantile(0.95, sum by (job, le) (rate(video_commerce_database_query_duration_seconds_bucket[5m])))
max by (consumergroup, topic) (kafka_consumergroup_lag)
```

If `promtool` is installed:

```bash
promtool test rules tests/monitoring/prometheus-rules.test.yml
```

Validate Alertmanager routing (the example webhook URL file must be mounted at
the `url_file` path, as Compose does):

```bash
docker run --rm --entrypoint /bin/amtool \
  -v "$PWD/monitoring:/m:ro" \
  -v "$PWD/monitoring/secrets/oncall-webhook-url.example:/etc/alertmanager/secrets/oncall-webhook-url:ro" \
  prom/alertmanager:v0.27.0 check-config /m/alertmanager.yml
```

## Multi-Worker Metrics

`gateway-api`, `recommendation-service`, `ranking-service` and
`interaction-ingest-service` run several uvicorn workers. They set
`PROMETHEUS_MULTIPROC_DIR` (a tmpfs in Compose, a `medium: Memory` emptyDir in
Helm), so every worker writes its metrics to that directory and `/metrics`
aggregates all of them, whichever worker answers the scrape. The other services
are single-process and do not set it; never put it in the shared environment,
or tests and one-shot jobs also switch to multiprocess mode.

- Startup runs `python -m video_commerce.common.prometheus_multiproc` before
  `exec uvicorn`. It removes `*.db` files left in the directory, because the
  emptyDir survives a container restart inside the same pod.
- Counters and histograms are summed over every worker that ever ran in the
  container, including exited ones, so they stay monotonic.
- Every Gauge declares `multiprocess_mode` (`tests/test_prometheus_multiprocess.py`
  fails otherwise): per-process counts such as in-flight requests, queue depth
  and RSS use `livesum`; per-process health flags use `livemin`/`livemax`;
  observations of shared state (Redis, outbox, release pointers) use
  `mostrecent`; versions and timestamps use `max`.
- Prometheus keeps one keep-alive connection, so one worker usually serves every
  scrape. Gauges refreshed at scrape time (Redis info, consumer lag, worker
  heartbeats, DB pool, Kafka producer state) reflect that worker. Process CPU
  and RSS are refreshed by each worker from its request path at most every 15
  seconds instead, so an idle worker reports its last value.
- `video_commerce_kafka_producer_connected` uses `livemax`: it reads 1 while
  any worker is connected, so `KafkaProducerDisconnected` fires only when no
  worker in the pod is connected.
- A worker removes its `live*` gauge files on clean shutdown. uvicorn has no
  child-exit hook, so a worker that crashes keeps its live gauges in the
  aggregate until the container restarts.
- `ranking-service` `/metrics` still forwards the coordinator's payload unless
  the Triton backend is active; only then are its own (aggregated) metrics
  exported.

Check that a counter is monotonic across workers under light load:

```bash
for i in $(seq 1 20); do
  docker compose exec -T gateway-api curl -sf http://127.0.0.1:8000/metrics \
    | awk '/^video_commerce_http_requests_total/ {s+=$2} END {print s}'
  curl -sf -o /dev/null http://localhost/health
done
```

The printed totals must never decrease.

## Grafana

Open `http://127.0.0.1:3000` and confirm:

- The `Prometheus` datasource is provisioned.
- The `Video Commerce Overview` dashboard appears under the `Video Commerce` folder.

## Jaeger

Generate a gateway-to-service trace:

```bash
curl -sS -H "x-api-key: $API_API_KEY" \
  -H "content-type: application/json" \
  -d '{"user_id":"trace-check","k":3,"context":{"source":"trace-validation"}}' \
  http://localhost/api/recommendations >/dev/null
```

Open `http://127.0.0.1:16686`, select `gateway-api`, and verify the trace includes a downstream `recommendation-service` span.

Generate a Kafka trace:

```bash
curl -sS -H "x-api-key: $API_API_KEY" \
  -H "content-type: application/json" \
  -d '{"user_id":"trace-check","product_id":"prod_1","action":"click","context":{"source":"trace-validation"}}' \
  http://localhost/api/interactions >/dev/null
```

In Jaeger, search for `interaction-ingest-service` and `feature-worker`; the worker consumer span should share trace context when the Kafka message headers include `traceparent`.
