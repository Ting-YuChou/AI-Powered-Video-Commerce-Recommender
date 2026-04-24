# Object Storage And Auth

## Object Storage
- `OBJECT_STORAGE_BACKEND=local` keeps the current shared-volume flow.
- `OBJECT_STORAGE_BACKEND=s3` uploads staged files to a durable S3-compatible bucket before enqueueing Kafka work.
- Gateway writes the durable `storage_path` into `content_jobs.storage_path`.
- Content workers materialize remote objects into `OBJECT_STORAGE_DOWNLOAD_DIR` before CLIP/OCR processing, then delete only the materialized temp file.
- S3 clients use bounded connect/read timeouts and standard retries. Set `OBJECT_STORAGE_CHECKSUM_ALGORITHM` and `OBJECT_STORAGE_SERVER_SIDE_ENCRYPTION` for checksum/SSE enforcement when the backend supports them.
- Bucket creation is intentionally conservative: startup may create the bucket only for a missing-bucket response, while auth, region, and permission errors fail startup instead of being swallowed.

## MinIO Bring-Up
- Start the profile with `docker compose --profile object-storage up -d minio minio-init`.
- Set:
  - `OBJECT_STORAGE_BACKEND=s3`
  - `OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000`
  - `OBJECT_STORAGE_BUCKET=video-commerce-assets`
  - `OBJECT_STORAGE_ACCESS_KEY_ID_FILE=/run/secrets/object_storage_access_key`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY_FILE=/run/secrets/object_storage_secret_key`
  - `OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS=5`
  - `OBJECT_STORAGE_READ_TIMEOUT_SECONDS=60`
  - `OBJECT_STORAGE_MAX_ATTEMPTS=3`
  - `OBJECT_STORAGE_CHECKSUM_ALGORITHM=SHA256`
  - `OBJECT_STORAGE_SERVER_SIDE_ENCRYPTION=AES256`

## Client Auth
- `SECURITY_AUTH_MODE=api_key` keeps the current `X-API-Key` flow.
- `SECURITY_AUTH_MODE=bearer` requires `Authorization: Bearer <jwt>` and `SECURITY_OIDC_ENABLED=true`.
- `SECURITY_AUTH_MODE=api_key_or_bearer` allows progressive migration.
- For local or staging JWT validation, use:
  - `SECURITY_JWT_SHARED_SECRET_FILE=/run/secrets/jwt_shared_secret`
  - `SECURITY_JWT_ALGORITHMS=HS256`
- For OIDC/JWKS validation, use:
  - `SECURITY_OIDC_ENABLED=true`
  - `SECURITY_OIDC_REQUIRED=true`
  - `SECURITY_OIDC_ISSUER=https://issuer.example.com/`
  - `SECURITY_OIDC_AUDIENCE=video-commerce-api`
  - `SECURITY_OIDC_JWKS_URL=https://issuer.example.com/.well-known/jwks.json`

## Secret Files
- Any environment variable can be sourced from `*_FILE`.
- Example:
  - `API_API_KEY_FILE=/run/secrets/public_api_key`
  - `SECURITY_INTERNAL_SERVICE_KEY_FILE=/run/secrets/internal_service_key`
  - `REDIS_PASSWORD_FILE=/run/secrets/redis_password`
  - `DATABASE_URL_FILE=/run/secrets/database_url`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY_FILE=/run/secrets/object_storage_secret_key`
- Production config rejects empty API keys, empty internal service keys, empty Redis passwords, and the default Postgres/MinIO credentials. Leave the direct env var empty when using the matching `*_FILE` variable.
