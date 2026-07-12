-- Leased, resumable operational state for daily PIT dataset production.

CREATE TABLE IF NOT EXISTS pit_materialization_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    cutoff_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    phase VARCHAR(32) NOT NULL DEFAULT 'export',
    attempts INTEGER NOT NULL DEFAULT 0,
    export_attempt INTEGER,
    flink_job_id VARCHAR(32),
    worker_id VARCHAR(255),
    lease_expires_at TIMESTAMPTZ,
    snapshot_id VARCHAR(64),
    manifest_uri TEXT,
    row_count BIGINT NOT NULL DEFAULT 0,
    quarantine_count BIGINT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_pit_materialization_runs_status_lease
ON pit_materialization_runs (status, lease_expires_at, cutoff_at);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pit_single_running_materialization
ON pit_materialization_runs (status)
WHERE status = 'running';

ALTER TABLE model_checkpoints
ADD COLUMN IF NOT EXISTS materialization_run_id VARCHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_checkpoints_pit_run
ON model_checkpoints (model_name, materialization_run_id)
WHERE materialization_run_id IS NOT NULL;

ALTER TABLE pit_materialization_runs
ADD COLUMN IF NOT EXISTS training_status VARCHAR(32);
ALTER TABLE pit_materialization_runs
ADD COLUMN IF NOT EXISTS training_worker_id VARCHAR(255);
ALTER TABLE pit_materialization_runs
ADD COLUMN IF NOT EXISTS training_lease_expires_at TIMESTAMPTZ;
ALTER TABLE pit_materialization_runs
ADD COLUMN IF NOT EXISTS trained_model_version VARCHAR(128);
