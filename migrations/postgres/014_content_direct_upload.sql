-- Direct uploads (DATA_DIRECT_UPLOAD_ENABLED).
--
-- POST /api/content/uploads reserves a content_jobs row in status
-- 'awaiting_upload' with no storage_path and no outbox row. The client uploads
-- straight to object storage with a presigned POST, then
-- POST /api/content/uploads/{content_id}/complete checks the object against the
-- declared size and type, copies it to raw/, and moves the row to
-- 'pending_publish' together with its outbox row. The content-task-publisher
-- marks reservations that were never completed as 'upload_expired'.
ALTER TABLE content_jobs
    ADD COLUMN IF NOT EXISTS declared_size_bytes BIGINT,
    ADD COLUMN IF NOT EXISTS declared_content_type VARCHAR(255),
    ADD COLUMN IF NOT EXISTS upload_expires_at TIMESTAMPTZ;

-- The expiry pass scans only open reservations.
CREATE INDEX IF NOT EXISTS ix_content_jobs_awaiting_upload
    ON content_jobs (upload_expires_at)
    WHERE status = 'awaiting_upload';
