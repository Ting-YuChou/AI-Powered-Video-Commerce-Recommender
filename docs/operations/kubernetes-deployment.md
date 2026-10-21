# Kubernetes Deployment

This guide covers the first Kubernetes deployment path for the split-service
runtime. The chart deploys only application workloads. Postgres, Redis, Kafka,
and S3-compatible object storage must be provided outside the chart.

## Architecture

- Public traffic enters through the Caddy Service and optional Ingress.
- Caddy serves the frontend and reverse proxies API traffic to `gateway-api`.
- Backend services run as independent Deployments:
  - `gateway-api`
  - `recommendation-service`
  - `interaction-ingest-service`
  - `ranking-service`
  - `ranking-coordinator`
  - `ranking-runner`
  - `content-worker`
  - optional GPU-backed `asr-service` for uploaded-video speech transcription
  - `feature-worker` (only when `appConfig.featurePipelineMode=python`; it is
    disabled by default because the Flink job owns official feature writes)
  - `model-trainer`
- `ranking-runner` uses a headless Service so `ranking-coordinator` can resolve
  individual pod endpoints through DNS.
- The coordinator and runner TCP ports are covered by NetworkPolicies
  (`networkPolicy.rankingRuntime`) and connection auth
  (`appConfig.ranking.runtimeAuthMode`). Runner drain/undrain is accepted only
  over loopback. See `docs/operations/ranking-runtime-auth.md`.
- Uploads, model artifacts, and checkpoints use S3-compatible object storage.
  Pod-local paths such as `/app/models`, `/app/uploads`, and
  `/tmp/object-storage` are `emptyDir` scratch space.

## Required External Services

Create or provision these before installing the chart:

- Postgres database and async SQLAlchemy URL. Its `max_connections` (or a
  PgBouncer `max_db_connections`) must cover `postgres.maxConnectionsBudget`;
  see `postgres-connection-budget.md` for the per-workload table.
- Redis for state and optionally a separate Redis for recommendation cache.
- Kafka brokers reachable from the cluster.
- S3-compatible object storage bucket.
- Flink (see below) for the default `featurePipelineMode: flink`.
- NVIDIA GPU nodes and a model-cache volume when `asr.enabled=true`.

Kafka topics must exist unless `kafkaTopicInitJob.enabled=true` is used. The
chart's `kafkaTopicInitJob.topics` is the full list:

- `user-interactions`
- `video-processing-tasks`
- `recommendation-events`
- `feature-updates`
- `feature-history-snapshots`
- `catalog-feature-events`
- `late-interaction-events`
- `dead-letter-events`
- backfill topics: `user-interactions-backfill`,
  `recommendation-events-backfill`, `feature-updates-backfill`,
  `catalog-feature-events-backfill`, `feature-history-backfill-quarantine`

### Flink (external)

The chart does not deploy Flink, but the default feature pipeline depends on it:
with `featurePipelineMode: flink` the Python feature worker is off and only the
Flink interaction job writes official Redis/Postgres features. Run the job
outside the chart (Flink Kubernetes operator or a session cluster) with this
contract:

- Job name `video-commerce-interaction-features`, exactly one non-terminal
  instance (the Compose submitter and alerts assume this).
- Reads `user-interactions` and `recommendation-events`; writes
  `feature-history-snapshots`, `late-interaction-events` and
  `dead-letter-events`, plus Redis features and Postgres views.
- Durable checkpoints and savepoints under runtime subdirectories (for example
  `s3://.../flink/checkpoints/runtime` and `.../savepoints/runtime`), restored
  on every resubmit; see `docs/operations/flink-realtime-recovery.md`.
- `FLINK_KAFKA_TRANSACTION_TIMEOUT_MS` within the broker transaction timeout.

Set `appConfig.flinkRestUrl` to the JobManager REST URL so the gateway's
`/readyz/full` reports the job state and last checkpoint age. It is reported
only and never fails readiness.

For production Kafka, prefer managing topics outside this chart so partition and
replication settings are reviewed with the platform team.

## Secrets

Use an existing Kubernetes Secret in production:

```yaml
secrets:
  create: false
  existingSecret: video-commerce-secrets
```

The default key names are:

- `API_API_KEY`
- `SECURITY_INTERNAL_SERVICE_KEY`
- `REDIS_PASSWORD`
- `REDIS_CACHE_PASSWORD`
- `DATABASE_URL`
- `OBJECT_STORAGE_ACCESS_KEY_ID`
- `OBJECT_STORAGE_SECRET_ACCESS_KEY`
- `SECURITY_JWT_SHARED_SECRET`

The chart also supports creating a Secret from `secrets.values.*`, but this is
intended only for local or ephemeral staging installs.

## Minimal Values

```yaml
images:
  backend:
    repository: registry.example.com/video-commerce-backend
    tag: "2026-05-16"
  frontend:
    repository: registry.example.com/video-commerce-edge
    tag: "2026-05-16"

secrets:
  create: false
  existingSecret: video-commerce-secrets

external:
  redis:
    host: redis-state.example.internal
  redisCache:
    host: redis-cache.example.internal
  kafka:
    bootstrapServers: kafka-0.example.internal:9092,kafka-1.example.internal:9092,kafka-2.example.internal:9092
  objectStorage:
    endpointUrl: https://s3.example.internal
    region: us-east-1
    bucket: video-commerce-assets

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: video-commerce.example.com
      paths:
        - path: /
          pathType: Prefix
```

## Optional Speech-To-Text Worker

Speech-to-text is isolated from the normal backend image. The content worker
extracts bounded 16 kHz mono audio and calls the internal ASR service through
the OpenAI-compatible transcription endpoint. ASR failures are degraded
content features: CLIP/OCR processing can still complete.

The ASR image also loads `Qwen/Qwen3-ForcedAligner-0.6B`. Successful responses
include bounded timestamped segments. If forced alignment fails after a valid
transcription, the transcript is retained and `alignment_status=degraded`; the
ranker's ASR presence mask remains false for that content item.

`Dockerfile.asr` runs a private multipart adapter around the `qwen-asr`
Python inference API. Do not replace it with the `qwen-asr-serve` command
without also updating the content-worker client contract; the upstream server
uses a chat-completions audio request format.

The adapter admits one GPU transcription at a time by default
(`asr.maxConcurrentTranscriptions`) and rejects requests that wait longer than
`asr.queueWaitSeconds` with `503`. The content worker can retry this
pre-inference capacity response, but it does not retry an ASR request after an
HTTP timeout because the original GPU inference may still be running.

Video processing remains inside the Kafka handler until features are stored.
Keep `external.kafka.consumerMaxPollIntervalMs` large enough for FFmpeg,
CLIP/OCR, and one ASR attempt; the chart default is `600000` milliseconds for
this offline worker path.

Build and publish the isolated GPU image from `Dockerfile.asr`, then enable
transcript capture without recommendation impact first:

```yaml
images:
  asr:
    repository: registry.example.com/video-commerce-asr
    tag: "2026-05-24"

appConfig:
  speechToText:
    enabled: true
    model: Qwen/Qwen3-ASR-0.6B
  recommendation:
    speechCategoryCandidatesEnabled: false

asr:
  enabled: true
  model: Qwen/Qwen3-ASR-0.6B
  alignerModel: Qwen/Qwen3-ForcedAligner-0.6B
  maxConcurrentTranscriptions: "1"
  queueWaitSeconds: "5"
  modelCache:
    existingClaim: video-commerce-asr-models
  nodeSelector:
    accelerator: nvidia-gpu
```

After transcript/alignment quality and `video_commerce_asr_transcriptions_total`,
`video_commerce_asr_alignment_total`, alignment latency, and segment-count
metrics are reviewed, enable
speech-derived category candidates:

```yaml
appConfig:
  recommendation:
    speechCategoryCandidatesEnabled: true
```

The ASR Service is internal-only. Do not log transcript text or copy it to
Kafka feature update events or content job payloads.

Install:

```bash
helm upgrade --install video-commerce ./charts/video-commerce \
  --namespace video-commerce --create-namespace \
  -f values.production.yaml
```

## Validation

Before applying:

```bash
helm lint charts/video-commerce
helm template video-commerce charts/video-commerce -f values.production.yaml >/tmp/video-commerce.yaml
kubeconform -strict /tmp/video-commerce.yaml
```

After applying:

```bash
kubectl -n video-commerce rollout status deploy/video-commerce-gateway-api
kubectl -n video-commerce rollout status deploy/video-commerce-recommendation-service
kubectl -n video-commerce rollout status deploy/video-commerce-interaction-ingest-service
kubectl -n video-commerce rollout status deploy/video-commerce-ranking-service
kubectl -n video-commerce rollout status deploy/video-commerce-ranking-coordinator
kubectl -n video-commerce rollout status deploy/video-commerce-ranking-runner
curl -fsS https://video-commerce.example.com/readyz
curl -fsS -H "x-api-key: $API_API_KEY" https://video-commerce.example.com/readyz/full
```

Readiness semantics:

- Caddy's readiness probe is its own `/caddy-healthz`, so the edge and the
  static frontend stay up while the gateway is degraded.
- Gateway `/readyz` is public, cached for `API_READINESS_CACHE_SECONDS`, and
  checks only the gateway itself (startup, auth configured, Redis). It does not
  probe shared dependencies, so a ranking or Kafka outage shows up as 503s on
  the affected routes instead of removing every gateway pod at once. Error
  details are redacted.
- Gateway `/readyz/full` needs credentials. It probes recommendation, ranking,
  ingest, Kafka, Postgres, object storage, worker heartbeats (feature-worker
  only in python mode) and, when configured, the Flink job. It returns 503 when
  any of them is unhealthy (degraded is allowed), so use it as the deep
  post-deploy check, not as a Kubernetes probe.
- Interaction ingest stays unready without Kafka or Postgres, because it only
  acknowledges events after the Kafka publish and the idempotency ledger.
  Gateway and ingest reconnect to Kafka in the background after a failed
  start, so a short broker outage at boot no longer needs a pod restart.
- A broker that goes away after ingest started is noticed per publish. Ingest
  waits at most `SERVICE_INTERACTION_INGEST_PUBLISH_TIMEOUT_MS` (3000 by
  default; config validation keeps it below `SERVICE_PROXY_READ_TIMEOUT_SECONDS`)
  and then answers a retryable 503 `KAFKA_PUBLISH_TIMEOUT` with
  `Retry-After: 1` and the `event_id`, instead of a gateway 504. After
  `SERVICE_INTERACTION_INGEST_BREAKER_FAILURE_THRESHOLD` (5) such failures in a
  row, or with more than `SERVICE_INTERACTION_INGEST_MAX_OUTSTANDING_PUBLISHES`
  (1000) sends unresolved per worker, it answers 503 `KAFKA_UNAVAILABLE` at
  once, letting one probe publish through every
  `SERVICE_INTERACTION_INGEST_BREAKER_COOLDOWN_SECONDS` (5). Once aiokafka fails
  a send as unreachable, the producer marks itself disconnected, which also
  makes ingest unready, and the supervisor reconnects it in the background.
- A timed-out send can still land, so its event_id stays pending, with its
  ledger lease renewed, until Kafka settles it. Retries get 503
  `INTERACTION_PUBLISH_PENDING` until then and 202 afterwards (with
  `duplicate: true` when the first send landed).

Then run a small smoke baseline against the Ingress URL:

```bash
python scripts/loadtest_api_baseline.py \
  --base-url https://video-commerce.example.com \
  --requests 500 \
  --concurrency 50 \
  --mode hot \
  --timeout 10
```

Do not claim production performance improvements until the relevant load test is
run and the result is recorded.
