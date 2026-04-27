# Observability Validation

Use this checklist after changing metrics, alerts, or tracing.

## Local Stack

```bash
docker compose config -q
docker compose up -d prometheus alertmanager grafana jaeger otel-collector
docker compose up -d gateway-api recommendation-service interaction-ingest-service content-worker feature-worker model-trainer
```

## Prometheus

Open `http://127.0.0.1:9090/targets` and confirm these jobs are up:

- `gateway-api`, `recommendation-service`, `interaction-ingest-service`
- `content-worker`, `feature-worker`, `model-trainer`
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
