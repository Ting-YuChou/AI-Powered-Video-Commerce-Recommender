# Object Storage And Auth

## Object Storage
- `OBJECT_STORAGE_BACKEND=local` keeps the current shared-volume flow.
- `OBJECT_STORAGE_BACKEND=s3` uploads staged files to a durable S3-compatible bucket before enqueueing Kafka work.
- Gateway writes the durable `storage_path` into `content_jobs.storage_path`.
- Content workers materialize remote objects into `OBJECT_STORAGE_DOWNLOAD_DIR` before CLIP/OCR processing, then delete only the materialized temp file.

## MinIO Bring-Up
- Start the profile with `docker compose --profile object-storage up -d minio minio-init`.
- Set:
  - `OBJECT_STORAGE_BACKEND=s3`
  - `OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000`
  - `OBJECT_STORAGE_BUCKET=video-commerce-assets`
  - `OBJECT_STORAGE_ACCESS_KEY_ID_FILE=/run/secrets/object_storage_access_key`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY_FILE=/run/secrets/object_storage_secret_key`

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
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY_FILE=/run/secrets/object_storage_secret_key`
