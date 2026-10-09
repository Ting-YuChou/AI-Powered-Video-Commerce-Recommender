CREATE TABLE IF NOT EXISTS model_releases (
    release_id VARCHAR(36) PRIMARY KEY,
    checkpoint_id BIGINT NOT NULL UNIQUE REFERENCES model_checkpoints(id),
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    lifecycle_state VARCHAR(32) NOT NULL DEFAULT 'registered',
    validation_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    bundle_manifest JSONB NOT NULL,
    bootstrap_uncompared BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS ix_model_releases_state
    ON model_releases (model_name, lifecycle_state, created_at DESC);

CREATE TABLE IF NOT EXISTS model_release_evaluations (
    evaluation_id VARCHAR(36) PRIMARY KEY,
    release_id VARCHAR(36) NOT NULL REFERENCES model_releases(release_id),
    champion_release_id VARCHAR(36) REFERENCES model_releases(release_id),
    dataset_manifest_uri TEXT NOT NULL,
    dataset_manifest_sha256 VARCHAR(64) NOT NULL,
    holdout_start TIMESTAMPTZ NOT NULL,
    holdout_end TIMESTAMPTZ NOT NULL,
    policy_version VARCHAR(128) NOT NULL,
    decision VARCHAR(32) NOT NULL,
    metrics JSONB NOT NULL,
    slice_metrics JSONB NOT NULL,
    bootstrap JSONB NOT NULL,
    gate_config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (release_id, champion_release_id, dataset_manifest_sha256, policy_version)
);

CREATE TABLE IF NOT EXISTS model_release_pointers (
    model_name VARCHAR(128) NOT NULL,
    environment VARCHAR(64) NOT NULL,
    slot VARCHAR(32) NOT NULL,
    release_id VARCHAR(36) NOT NULL REFERENCES model_releases(release_id),
    generation BIGINT NOT NULL DEFAULT 1,
    updated_by VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_name, environment, slot),
    UNIQUE (model_name, environment, slot)
);

CREATE TABLE IF NOT EXISTS model_release_transitions (
    id BIGSERIAL PRIMARY KEY,
    release_id VARCHAR(36) NOT NULL REFERENCES model_releases(release_id),
    evaluation_id VARCHAR(36) REFERENCES model_release_evaluations(evaluation_id),
    from_state VARCHAR(32),
    to_state VARCHAR(32) NOT NULL,
    actor VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    pointer_generation BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_model_release_transitions_release
    ON model_release_transitions (release_id, created_at DESC);
