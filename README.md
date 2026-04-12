# AI-Powered Video Commerce Recommender

Production-oriented single-VM deployment for a video commerce recommendation system built as a microservice stack:

- `caddy` serves the frontend and terminates public traffic.
- `gateway-api` is the only HTTP entrypoint for backend routes.
- `recommendation-service` handles online recommendation serving.
- `interaction-ingest-service` accepts interaction writes and publishes Kafka events.
- `content-worker`, `feature-worker`, and `model-trainer` process async work.
- `redis`, `postgres`, and `kafka` back the runtime data plane.

`app.py` is no longer the production entrypoint. The supported production path is Docker Compose plus the service modules above.

## Quick Start

### Prerequisites

- Docker Engine with `docker compose`
- At least 8 GB RAM
- FFmpeg/Tesseract are already baked into the backend image; you do not need them on the host

### Start The Stack

```bash
cp .env.example .env 2>/dev/null || true
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

## Production Defaults

- Only `caddy` exposes host ports.
- Internal services authenticate with `X-Internal-Service-Key`.
- Redis is used for cache, online features, and worker heartbeats.
- Kafka is the only async event bus.
- Postgres stores durable interaction events, content jobs, product catalog snapshots, and model checkpoints.
- Uploads are streamed to disk and queued to Kafka; they no longer read the entire file into memory.
- `*_FILE` environment variables are supported for secret injection.
- Optional `OBJECT_STORAGE_BACKEND=s3` enables durable upload persistence via an S3-compatible backend.

## Environment Variables

Important overrides for production:

```bash
API_API_KEY=replace-with-client-api-key
SECURITY_INTERNAL_SERVICE_KEY=replace-with-internal-service-secret
DATABASE_URL=postgresql+asyncpg://video_commerce:video_commerce@postgres:5432/video_commerce
DATA_MAX_FILE_SIZE=104857600
CADDY_SITE_ADDRESS=your.domain.com
GRAFANA_ADMIN_USER=grafana
GRAFANA_ADMIN_PASSWORD=replace-with-strong-password
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000
OBJECT_STORAGE_BUCKET=video-commerce-assets
SECURITY_AUTH_MODE=api_key_or_bearer
SECURITY_OIDC_ENABLED=true
SECURITY_OIDC_ISSUER=https://issuer.example.com/
SECURITY_OIDC_AUDIENCE=video-commerce-api
SECURITY_OIDC_JWKS_URL=https://issuer.example.com/.well-known/jwks.json
```

## Service Layout

- `gateway_api.py`
- `recommendation_api.py`
- `interaction_ingest_api.py`
- `kafka_workers/video_processor.py`
- `kafka_workers/feature_updater.py`
- `model_trainer.py`
- `system_store.py`

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
- `monitoring/prometheus.yml` defines Prometheus scrape targets.
- `monitoring/prometheus-rules.yml` defines alert rules loaded by Prometheus.
- `Code.Frontend/Dockerfile` builds the frontend and packages it behind Caddy.
- Operational docs:
  - `docs/operations/object-storage-and-auth.md`
  - `docs/operations/slo-and-alerting.md`
  - `docs/operations/deploy-rollback-runbook.md`
  - `docs/operations/loadtest-baseline.md`
