# SLO And Alerting

## Initial SLOs
- Availability: gateway 5xx rate under 1% rolling 30 days.
- Latency: gateway `p95 < 1000ms`, `p99 < 2000ms` for `/api/recommendations`.
- Freshness: worker heartbeat present for `content-worker`, `feature-worker`, and `model-trainer`.
- Durability: uploads must reach either local shared storage or S3-compatible storage before Kafka enqueue returns success.

## Alert Rules
Prometheus now loads `monitoring/prometheus-rules.yml` and sends alerts to Alertmanager.

Current alerts:
- `GatewayErrorBudgetBurn`
- `GatewayP95LatencyHigh`
- `GatewayUnavailable`
- `RecommendationServiceUnavailable`
- `InteractionIngestServiceUnavailable`
- `WorkerHeartbeatMissing`
- `KafkaProducerDisconnected`
- `RedisMemoryHigh`

## Operator Response
- Gateway 5xx or latency alerts: inspect `gateway-api` logs and `gateway /readyz`.
- Dependency down alerts: inspect `docker compose ps` and `docker compose logs <service>`.
- Worker heartbeat missing: inspect Redis, Kafka lag, and the worker logs; verify `video_commerce_worker_live_instances`.
- Kafka producer disconnected: verify broker health and topic availability before re-enabling traffic.
