CREATE TABLE IF NOT EXISTS model_release_instance_observations (
    model_name VARCHAR(128) NOT NULL,
    environment VARCHAR(64) NOT NULL,
    service_name VARCHAR(128) NOT NULL,
    instance_id VARCHAR(255) NOT NULL,
    desired_release_id VARCHAR(36),
    desired_generation BIGINT NOT NULL DEFAULT 0,
    desired_bundle_digest VARCHAR(64),
    observed_release_id VARCHAR(36),
    observed_generation BIGINT,
    observed_model_version VARCHAR(128),
    bundle_digest VARCHAR(64),
    state VARCHAR(32) NOT NULL,
    last_error TEXT,
    activated_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_name, environment, service_name, instance_id)
);

CREATE INDEX IF NOT EXISTS ix_model_release_observations_heartbeat
    ON model_release_instance_observations
    (model_name, environment, heartbeat_at DESC);

CREATE TABLE IF NOT EXISTS iceberg_maintenance_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    mode VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    owner_token VARCHAR(64),
    lease_expires_at TIMESTAMPTZ,
    plan_sha256 VARCHAR(64),
    head_snapshot_id VARCHAR(64),
    plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    before_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    after_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    deleted_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_iceberg_maintenance_runs_table_status
    ON iceberg_maintenance_runs (table_name, status, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_iceberg_maintenance_active_table
    ON iceberg_maintenance_runs (table_name)
    WHERE status = 'running';
