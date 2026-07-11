-- Operational catalog activation and transactional outbox state.
-- These tables provide reliable Kafka transport; they are not an offline store.

CREATE TABLE IF NOT EXISTS product_catalog_snapshot (
    product_id VARCHAR(255) PRIMARY KEY,
    snapshot JSON NOT NULL DEFAULT '{}'::json,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE product_catalog_snapshot
    ADD COLUMN IF NOT EXISTS activation_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_version VARCHAR(255);

CREATE INDEX IF NOT EXISTS ix_product_catalog_snapshot_activation_id
ON product_catalog_snapshot (activation_id);

CREATE TABLE IF NOT EXISTS catalog_activations (
    activation_id VARCHAR(64) PRIMARY KEY,
    source_version VARCHAR(255) NOT NULL UNIQUE,
    expected_count INTEGER NOT NULL,
    manifest_hash VARCHAR(64) NOT NULL,
    actual_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'staging',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

ALTER TABLE catalog_activations
    ADD COLUMN IF NOT EXISTS manifest_hash VARCHAR(64);

CREATE TABLE IF NOT EXISTS catalog_feature_outbox (
    event_id VARCHAR(64) PRIMARY KEY,
    activation_id VARCHAR(64) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    payload JSON NOT NULL DEFAULT '{}'::json,
    payload_hash VARCHAR(64) NOT NULL,
    event_payload JSON NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_by VARCHAR(255),
    claim_expires_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_catalog_feature_outbox_activation
        FOREIGN KEY (activation_id) REFERENCES catalog_activations (activation_id)
);

CREATE INDEX IF NOT EXISTS ix_catalog_feature_outbox_activation_id
ON catalog_feature_outbox (activation_id);

CREATE INDEX IF NOT EXISTS ix_catalog_feature_outbox_product_id
ON catalog_feature_outbox (product_id);

CREATE INDEX IF NOT EXISTS ix_catalog_feature_outbox_pending
ON catalog_feature_outbox (published_at, claim_expires_at, created_at);
