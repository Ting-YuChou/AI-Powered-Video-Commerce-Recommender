# Object Storage And Auth

## Object Storage
- `OBJECT_STORAGE_BACKEND=local` keeps the current shared-volume flow.
- `OBJECT_STORAGE_BACKEND=s3` uploads staged files to a durable S3-compatible bucket before enqueueing Kafka work.
- Gateway writes the durable `storage_path` into `content_jobs.storage_path`.
- Content workers materialize remote objects into `OBJECT_STORAGE_DOWNLOAD_DIR` before CLIP/OCR processing, then delete only the materialized temp file.
- S3 clients use bounded connect/read timeouts and standard retries. Set `OBJECT_STORAGE_CHECKSUM_ALGORITHM` and `OBJECT_STORAGE_SERVER_SIDE_ENCRYPTION` for checksum/SSE enforcement when the backend supports them.
- Bucket creation is intentionally conservative: startup may create the bucket only for a missing-bucket response, while auth, region, and permission errors fail startup instead of being swallowed.

## Key Layout
In S3 mode, keys under `OBJECT_STORAGE_PREFIX` (default `uploads`) are:
- `<prefix>/raw/<content_id><suffix>`: uploaded videos. Direct uploads add a
  random part: `<prefix>/raw/<content_id>-<random><suffix>`.
- `<prefix>/incoming/<content_id><suffix>`: direct uploads before /complete
  copies them to `raw/` (see
  [Direct uploads](event-and-artifact-reliability.md#direct-uploads-migration-014)).
- `<prefix>/artifacts/...`: model artifacts.
- `<prefix>/content-features/...`: content feature artifacts.

Uploads written before the `raw/` layout stay at `<prefix>/<content_id><suffix>`
and are not moved. `content_jobs.storage_path` holds the full URI, so workers
read both the same way. Local mode is unchanged: uploads stay flat in
`DATA_UPLOAD_DIR`.

All keys stay under `<prefix>/`. Object permissions on `<prefix>/*` cover
`raw/` and `incoming/`. Listing is different: the orphan reconciler lists both `<prefix>/raw/`
and `<prefix>/`, so an `s3:ListBucket` condition on `s3:prefix` must allow
both. `StringLike` `<prefix>/*` does; `StringEquals` `<prefix>/` does not.

Direct uploads need these on the gateway's credentials, because a presigned
POST acts with the permissions of whoever signed it:
- `s3:PutObject` on `<prefix>/incoming/*` (the browser's POST) and on
  `<prefix>/raw/*` (the copy).
- `s3:GetObject` on `<prefix>/incoming/*` (the HEAD and the copy source).
- `s3:DeleteObject` on `<prefix>/incoming/*` and `<prefix>/raw/*` (cleanup).

Set `OBJECT_STORAGE_PUBLIC_ENDPOINT_URL` when browsers reach storage at a
different address than the gateway does, for example `http://127.0.0.1:9000`
for Compose MinIO while the gateway uses `http://minio:9000`. Presigned POSTs
are signed for that endpoint. It is read only when the gateway signs a direct
upload, and validated only while `DATA_DIRECT_UPLOAD_ENABLED=true`. The bucket also needs a CORS rule that allows
`POST` from the frontend origin and exposes `ETag`; the frontend change that
uses direct uploads ships it.

## Lifecycle
`deploy/object-storage/lifecycle.json` holds the bucket lifecycle rules, in the
S3 API format, for AWS S3. Whoever manages the bucket applies it; the services
never change bucket configuration.
- **Raw uploads** (`uploads/raw/`): move to `STANDARD_IA` after 30 days and are
  kept forever.
  - Do not add an expiration rule. A queued job whose object is gone fails as
    `failed_missing_object`, and reprocessing
    (`scripts/backfill_temporal_multimodal.py`) reads the raw video again.
  - `STANDARD_IA` charges per GB retrieved and bills at least 30 days.
    Reprocessing old uploads pays that retrieval charge.
  - `STANDARD_IA` is an AWS storage class. Most S3-compatible stores (MinIO,
    Ceph RGW) do not have it unless an operator defines it, so they reject or
    ignore this rule. On those, keep the `incoming/` and multipart rules.
- **Direct uploads before /complete** (`uploads/incoming/`): deleted after 1
  day. /complete copies the file to `raw/` and deletes the incoming object, so
  this only removes uploads that were never completed, or whose delete failed.
  `DATA_DIRECT_UPLOAD_URL_TTL_SECONDS` plus
  `DATA_DIRECT_UPLOAD_COMPLETE_GRACE_SECONDS` must stay under a day so the
  rule never removes an upload that can still be completed; config validation
  enforces it.
- **Incomplete multipart uploads** (whole bucket): aborted after 7 days. The
  gateway uploads large files in parts. boto3 aborts the upload when it fails,
  but a gateway killed mid-upload (eviction, OOM) leaves its parts behind,
  billed but invisible in listings.
- **Applying it:**
  1. Read the current rules. `put-bucket-lifecycle-configuration` replaces the
     bucket's whole lifecycle configuration, so anything not in the file you
     put is deleted. The read fails with `NoSuchLifecycleConfiguration` when
     the bucket has no rules yet.

     ```bash
     aws s3api get-bucket-lifecycle-configuration --bucket <bucket> > current-lifecycle.json
     ```
  2. Write `merged-lifecycle.json`: the current rules plus the three rules from
     the repo file, with `uploads/` replaced by your `OBJECT_STORAGE_PREFIX`.
  3. Put the merged file:

     ```bash
     aws s3api put-bucket-lifecycle-configuration --bucket <bucket> --lifecycle-configuration file://merged-lifecycle.json
     ```

  Add `--endpoint-url` for a non-AWS endpoint.
- **Legacy uploads** do not match the `raw/` rule and stay in `STANDARD`. A
  lifecycle filter cannot select only the direct children of a prefix.
- **MinIO (Compose)** gets no rules yet. Community MinIO has no `STANDARD_IA`
  class (transitions need a remote tier), and it removes stale multipart
  uploads itself. Compose does not enable direct uploads; the change that
  wires them into Compose also adds the `incoming/` expiry to `minio-init`.

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
- `SECURITY_AUTH_MODE=disabled` is the local Compose default (`.env.example`).
  Production config rejects it.
- The gateway fails closed. In `api_key` or `api_key_or_bearer` mode with no
  API key (and no OIDC validator), protected routes return
  `503 AUTH_NOT_CONFIGURED` instead of allowing anonymous access.
- Compose binds Caddy to `EDGE_BIND_ADDRESS` (default `127.0.0.1`). Set
  `EDGE_BIND_ADDRESS=0.0.0.0` only after configuring a key or OIDC and a
  non-disabled auth mode.
- Known gap: the bundled UI does not send an API key or bearer token yet. A
  public edge with auth enabled needs an authenticated frontend or a
  separate UI access path (review item 22).
- For local or staging JWT validation, use:
  - `SECURITY_JWT_SHARED_SECRET_FILE=/run/secrets/jwt_shared_secret`
  - `SECURITY_JWT_ALGORITHMS=HS256`
- For OIDC/JWKS validation, use:
  - `SECURITY_OIDC_ENABLED=true`
  - `SECURITY_OIDC_REQUIRED=true`
  - `SECURITY_OIDC_ISSUER=https://issuer.example.com/`
  - `SECURITY_OIDC_AUDIENCE=video-commerce-api`
  - `SECURITY_OIDC_JWKS_URL=https://issuer.example.com/.well-known/jwks.json`
  - Production rejects bearer auth without issuer, audience and a JWKS URL,
    and rejects `SECURITY_JWT_SHARED_SECRET`.
- Bearer tokens must carry `exp`. JWKS lookups run off the event loop
  (`SECURITY_JWKS_TIMEOUT_SECONDS`, default 2), and unknown `kid`s force at most
  one JWKS refresh per `SECURITY_JWKS_MIN_REFRESH_INTERVAL_SECONDS` (30).
- Bearer identity scope (server side only; API-key callers are treated as a
  trusted backend and keep sending `user_id`):
  - Recommendation, interaction and upload requests must use the token's own
    user: `user_id` must equal the `SECURITY_OIDC_USER_ID_CLAIM` claim
    (default `sub`), otherwise `403 USER_MISMATCH`. An upload without
    `user_id` is attributed to the subject.
  - `GET /api/content/{id}/status` returns 404 for another user's content.
  - `GET /api/analytics` requires `SECURITY_ADMIN_SCOPE`
    (default `video-commerce:admin`) in the token's `scope`/`scp`.
  - Uploads at `priority=high` also need `SECURITY_ADMIN_SCOPE`. Without it
    the gateway lowers `high` to `normal` (no error) and logs
    `content_upload_priority_lowered` once per new job. API-key callers and
    `SECURITY_AUTH_MODE=disabled` keep the requested priority. When a request
    carries both a valid bearer token and an API key, the bearer token decides.
  - Scopes are read from both `scope` and `scp`, each a space-separated string
    or a list of strings. Other claim shapes grant no scope.
  - `200` upload responses (new or replayed) and `503 CONTENT_ENQUEUE_PENDING`
    return `effective_priority`; `409` responses do not. A replay reports the
    priority the job was stored with.
  - The stored job, the Kafka task, the `Idempotency-Key` fingerprint and
    `video_commerce_content_uploads_total` use the effective priority, so
    retrying a lowered upload with the same key replays it whether the retry
    asks for `high` or `normal`. A replay also matches the fingerprint of the
    requested priority, so keys first used before this rule existed, on an old
    gateway pod during a rollout, or by a caller whose scope has since changed
    still replay instead of returning `409`.
- Public IDs are validated: `user_id` allows only `A-Z a-z 0-9 . _ @ -` (no
  `:` or glob characters, since it is embedded in Redis key patterns), and
  product/content IDs additionally allow `: / +`. `context` objects are limited
  to 8KB serialized, 64 keys and 3 levels of nesting.

## Rate Limits And Request Size
- The gateway buckets requests by bearer subject, then API key, then client
  IP. `X-Forwarded-For` is used only when the direct peer is in
  `API_TRUSTED_PROXY_CIDRS` (Caddy), taking the right-most untrusted address.
- Rate limiting fails open: a Redis call slower than `API_RATE_LIMIT_TIMEOUT_MS`
  (50) or failing is allowed, counted in
  `video_commerce_rate_limiter_errors_total{reason}`, and Redis is skipped for
  one second.
- Failed authentication is limited per client IP
  (`API_UNAUTHENTICATED_RATE_LIMIT_REQUESTS` per minute, then
  `429 AUTH_RATE_LIMITED`).
- Bodies are checked from `Content-Length` before parsing: uploads may be
  `DATA_MAX_FILE_SIZE` plus 1MiB and must send `Content-Length` (411
  otherwise); other requests are limited to `API_MAX_REQUEST_SIZE` (1MiB).
  Caddy enforces `EDGE_MAX_UPLOAD_BODY` (501MiB) and `EDGE_MAX_JSON_BODY`
  (1MiB) for chunked bodies. In Helm, the `tmp`, `uploads` and
  `objectStorage` scratch volumes have a 10Gi `sizeLimit`.
- `DATA_MAX_FILE_SIZE` is 100MiB in the code default, `.env.example`, Compose
  and Helm (a test keeps them equal). `EDGE_MAX_UPLOAD_BODY` is only the outer
  ceiling; raise both together if larger uploads are needed.
- `GET /api/content/upload-limits` returns `max_file_size`,
  `allowed_extensions` and `allowed_mime_types`. The frontend validates files
  against it. Until the call succeeds it checks types against built-in
  defaults and skips the size check (the gateway still enforces it), and it
  retries the call when an upload starts. Files the browser reports with an
  empty type are sent with the MIME type implied by their extension.
- The frontend sends one `Idempotency-Key` per selected file and reuses it on
  retries. It treats `503 CONTENT_ENQUEUE_PENDING` as accepted (the upload is
  durable) and keeps polling the returned `content_id`; `failed` and
  `failed_missing_object` end polling. Running the same file at a different
  priority uses a new key (the fingerprint includes priority), and a `409`
  for a failed job or a reused key is retried once with a new key.

## Secret Files
- Any environment variable can be sourced from `*_FILE`.
- Example:
  - `API_API_KEY_FILE=/run/secrets/public_api_key`
  - `SECURITY_INTERNAL_SERVICE_KEY_FILE=/run/secrets/internal_service_key`
  - `REDIS_PASSWORD_FILE=/run/secrets/redis_password`
  - `DATABASE_URL_FILE=/run/secrets/database_url`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY_FILE=/run/secrets/object_storage_secret_key`
- Production config rejects empty API keys, empty internal service keys, empty Redis passwords, and the default Postgres/MinIO credentials. Leave the direct env var empty when using the matching `*_FILE` variable.
- `SECURITY_INTERNAL_SERVICE_KEY` also authenticates the ranking coordinator/runner TCP protocol (`RANKING_RUNTIME_AUTH_MODE`, default `permissive`; production rejects `disabled`). See `docs/operations/ranking-runtime-auth.md` for the flip to `required`.
