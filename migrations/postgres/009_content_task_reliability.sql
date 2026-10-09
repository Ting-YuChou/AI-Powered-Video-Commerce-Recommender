ALTER TABLE content_jobs
    ADD COLUMN IF NOT EXISTS pipeline_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS task_event_id VARCHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS ux_content_jobs_task_event_id
    ON content_jobs (task_event_id)
    WHERE task_event_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS content_task_outbox (
    event_id VARCHAR(64) PRIMARY KEY,
    content_id VARCHAR(64) NOT NULL,
    pipeline_version VARCHAR(128) NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    event_payload JSONB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_by VARCHAR(255),
    claim_expires_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    terminal_at TIMESTAMPTZ,
    terminal_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_content_task_outbox_job
        FOREIGN KEY (content_id) REFERENCES content_jobs (content_id)
);

CREATE INDEX IF NOT EXISTS ix_content_task_outbox_pending
    ON content_task_outbox (published_at, terminal_at, next_attempt_at, claim_expires_at);

CREATE INDEX IF NOT EXISTS ix_content_task_outbox_content_id
    ON content_task_outbox (content_id);

CREATE TABLE IF NOT EXISTS content_processing_runs (
    content_id VARCHAR(64) NOT NULL,
    pipeline_version VARCHAR(128) NOT NULL,
    task_event_id VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    lease_owner VARCHAR(64),
    lease_expires_at TIMESTAMPTZ,
    artifact_uri TEXT,
    artifact_sha256 VARCHAR(64),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, pipeline_version)
);

CREATE INDEX IF NOT EXISTS ix_content_processing_runs_lease
    ON content_processing_runs (status, lease_expires_at);
