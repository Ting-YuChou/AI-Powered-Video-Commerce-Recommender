-- Stage B WS-4: idempotent content uploads and bounded processing redrives.
--
-- Upload retries that carry the same Idempotency-Key for the same user resolve to
-- the original content job. The request fingerprint is "sha256:<hex>:<size>" of
-- the streamed upload body, so a reused key with different bytes is rejected (409).
ALTER TABLE content_jobs
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128),
    ADD COLUMN IF NOT EXISTS request_fingerprint VARCHAR(128);

-- user_id is nullable for anonymous uploads; COALESCE keeps anonymous keys unique too.
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_jobs_user_idempotency_key
    ON content_jobs ((COALESCE(user_id, '')), idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- The orphan reconciler checks pages of listed objects with batched IN queries.
CREATE INDEX IF NOT EXISTS ix_content_jobs_storage_path
    ON content_jobs (storage_path);

-- The content-task-publisher reaper re-publishes the outbox row of a run whose
-- processing lease expired (worker died mid-run) and stops after
-- SERVICE_CONTENT_PROCESSING_MAX_REDRIVES redrives.
ALTER TABLE content_processing_runs
    ADD COLUMN IF NOT EXISTS redrive_count INTEGER NOT NULL DEFAULT 0;
