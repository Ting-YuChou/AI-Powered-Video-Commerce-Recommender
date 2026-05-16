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
  - `feature-worker`
  - `model-trainer`
- `ranking-runner` uses a headless Service so `ranking-coordinator` can resolve
  individual pod endpoints through DNS.
- Uploads, model artifacts, and checkpoints use S3-compatible object storage.
  Pod-local paths such as `/app/models`, `/app/uploads`, and
  `/tmp/object-storage` are `emptyDir` scratch space.

## Required External Services

Create or provision these before installing the chart:

- Postgres database and async SQLAlchemy URL.
- Redis for state and optionally a separate Redis for recommendation cache.
- Kafka brokers reachable from the cluster.
- S3-compatible object storage bucket.

Kafka topics must exist unless `kafkaTopicInitJob.enabled=true` is used:

- `user-interactions`
- `video-processing-tasks`
- `recommendation-events`
- `feature-updates`
- `dead-letter-events`

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
curl -fsS https://video-commerce.example.com/readyz/full
```

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
