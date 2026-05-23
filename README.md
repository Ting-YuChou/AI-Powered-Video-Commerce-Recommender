# AI-Powered Video Commerce Recommender

Production-oriented video commerce recommendation system built as a split-service microservice stack. The repository supports:

- Docker Compose for the production-like single-VM target.
- Helm/Kubernetes for multi-node application deployments with external managed stateful services.

- `caddy` serves the frontend and terminates public traffic.
- `gateway-api` is the only HTTP entrypoint for backend routes.
- `recommendation-service` handles online recommendation serving.
- `interaction-ingest-service` accepts interaction writes and publishes Kafka events.
- `content-worker`, `feature-worker`, and `model-trainer` process async work.
- `redis`, `postgres`, and `kafka` back the runtime data plane.

`video_commerce/services/legacy_app/app.py` is no longer the production entrypoint. Use the split service modules above through Docker Compose or the Helm chart.

## Quick Start

The fastest local path is Docker Compose.

### Prerequisites

- Docker Engine with `docker compose`
- At least 8 GB RAM
- FFmpeg/Tesseract are already baked into the backend image; you do not need them on the host

### Start The Stack

```bash
cp .env.example .env 2>/dev/null || true
# Fill production secrets in .env before starting:
# API_API_KEY, SECURITY_INTERNAL_SERVICE_KEY, REDIS_PASSWORD, POSTGRES_PASSWORD
chmod +x startup.sh
./startup.sh start
./startup.sh status
./startup.sh health
```

The default public edge URL is `http://localhost`. Set `CADDY_SITE_ADDRESS=your.domain.com` before starting the stack if you want Caddy to obtain TLS certificates for a real hostname.

### Core URLs

- Public edge: `http://localhost`
- Gateway health: `http://localhost/health`
- Gateway readiness: `http://localhost/readyz`
- API docs: `http://localhost/docs`
- Grafana: `http://127.0.0.1:3000`
- Prometheus: `http://127.0.0.1:9090`
- Jaeger traces: `http://127.0.0.1:16686`

Observability validation steps live in `docs/operations/observability-validation.md`.

## Kubernetes Deployment

The Helm chart lives in `charts/video-commerce/`. It deploys the application
services only; Postgres, Redis, Kafka, and S3-compatible object storage are
expected to be external managed services.

Required Kubernetes production assumptions:

- `OBJECT_STORAGE_BACKEND=s3`; pod-local `/app/models`, `/app/uploads`, and
  `/tmp/object-storage` are scratch `emptyDir` volumes.
- Caddy remains the only public edge service.
- `ranking-runner` uses a headless Service so `ranking-coordinator` can resolve
  runner pod endpoints through DNS.
- Kafka topics must exist before startup unless the optional
  `kafkaTopicInitJob.enabled=true` path is intentionally used.

Basic validation:

```bash
docker run --rm -v "$PWD:/work" -w /work alpine/helm:3.14.0 lint charts/video-commerce
docker run --rm -v "$PWD:/work" -w /work alpine/helm:3.14.0 template video-commerce charts/video-commerce >/tmp/video-commerce.yaml
docker run --rm -v /tmp:/work ghcr.io/yannh/kubeconform:latest -strict -summary /work/video-commerce.yaml
```

Deployment details and production values examples live in
`docs/operations/kubernetes-deployment.md`.

## Production Defaults

- Only `caddy` exposes host ports.
- Internal services authenticate with `X-Internal-Service-Key`.
- Redis is used for cache, online features, and worker heartbeats.
- Kafka is the only async event bus.
- Postgres stores durable interaction events, content jobs, product catalog snapshots, and model checkpoints.
- Uploads are streamed to disk and queued to Kafka; they no longer read the entire file into memory.
- `*_FILE` environment variables are supported for secret injection.
- Optional `OBJECT_STORAGE_BACKEND=s3` enables durable upload persistence via an S3-compatible backend.
- Production startup fails fast when API/internal keys, Redis password, or non-default Postgres credentials are missing.
- Kubernetes multi-node deployments require external Postgres, Redis, Kafka,
  and S3-compatible object storage; the Helm chart does not install those
  stateful services.

## Environment Variables

Important overrides for production:

```bash
API_API_KEY=replace-with-client-api-key
SECURITY_INTERNAL_SERVICE_KEY=replace-with-internal-service-secret
REDIS_PASSWORD=replace-with-redis-password
POSTGRES_PASSWORD=replace-with-postgres-password
DATABASE_URL=postgresql+asyncpg://video_commerce:replace-with-postgres-password@postgres:5432/video_commerce
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=5
DATA_MAX_FILE_SIZE=104857600
CADDY_SITE_ADDRESS=your.domain.com
GRAFANA_ADMIN_USER=grafana
GRAFANA_ADMIN_PASSWORD=replace-with-strong-password
MONITORING_ENABLE_TRACING=true
MONITORING_TRACING_SAMPLE_RATE=1.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000
OBJECT_STORAGE_BUCKET=video-commerce-assets
OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS=5
OBJECT_STORAGE_READ_TIMEOUT_SECONDS=60
OBJECT_STORAGE_MAX_ATTEMPTS=3
OBJECT_STORAGE_CHECKSUM_ALGORITHM=SHA256
OBJECT_STORAGE_SERVER_SIDE_ENCRYPTION=AES256
SECURITY_AUTH_MODE=api_key_or_bearer
SECURITY_OIDC_ENABLED=true
SECURITY_OIDC_ISSUER=https://issuer.example.com/
SECURITY_OIDC_AUDIENCE=video-commerce-api
SECURITY_OIDC_JWKS_URL=https://issuer.example.com/.well-known/jwks.json
```

## Service Layout

- `video_commerce/services/gateway/api.py`
- `video_commerce/services/recommendation/api.py`
- `video_commerce/services/interaction_ingest/api.py`
- `video_commerce/services/ranking_service/proxy_asgi.py`
- `video_commerce/services/ranking_coordinator/main.py`
- `video_commerce/services/ranking_runner/main.py`
- `video_commerce/services/content_worker/video_processor.py`
- `video_commerce/services/feature_worker/feature_updater.py`
- `video_commerce/services/model_trainer/main.py`
- `video_commerce/data_plane/system_store.py`

## Repository Layout

- `docs/architecture/`: system design notes.
- `docs/data/`: dataset, BigQuery, and CSV ingestion guides.
- `docs/ml/`: model training guides.
- `docs/operations/`: deployment and production operations docs.
- `video_commerce/common/`: shared config, schemas, auth, service helpers, telemetry, and observability.
- `video_commerce/data_plane/`: Kafka, Redis feature store, Postgres system store, and object storage integrations.
- `video_commerce/ml/`: content processing, recommendation, ranking, vector search, and model artifact code.
- `video_commerce/ranking_runtime/`: ranking batching, payload, coordinator, and runner protocol helpers.
- `video_commerce/services/*/ARCHITECTURE.md`: per-service responsibilities, interfaces, dependencies, and startup commands.
- `scripts/load_dataset.py`: local dataset loading utility.
- `data/embeddings/`: local embedding artifacts such as `video_embeddings_128d.npy`.

## Tests And CI

```bash
pip install -r requirements.txt
pytest
docker compose --profile test run --rm backend-tests
```

GitHub Actions now expects:

- backend unit tests
- backend tests inside Docker
- API contract tests
- frontend build and lint
- Docker compose smoke validation

## Operational Commands

```bash
./startup.sh start
./startup.sh stop
./startup.sh restart
./startup.sh logs gateway-api
./startup.sh health
```

## Notes

- `docker-compose.yml` is the production deployment source of truth for the single-VM target.
- `charts/video-commerce/` is the Helm deployment source of truth for the
  multi-node Kubernetes application target.
- `monitoring/prometheus.yml` defines Prometheus scrape targets.
- `monitoring/prometheus-rules.yml` defines alert rules loaded by Prometheus.
- `Code.Frontend/Dockerfile` builds the frontend and packages it behind Caddy.
- Operational docs:
  - `docs/operations/kubernetes-deployment.md`
  - `docs/operations/object-storage-and-auth.md`
  - `docs/operations/slo-and-alerting.md`
  - `docs/operations/deploy-rollback-runbook.md`
  - `docs/operations/loadtest-baseline.md`
