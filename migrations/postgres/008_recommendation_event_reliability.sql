CREATE TABLE IF NOT EXISTS recommendation_impression_views (
    event_id VARCHAR(64) PRIMARY KEY,
    impression_id VARCHAR(64) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    position INTEGER NOT NULL,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    viewed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_recommendation_impression_views_impression_product
    ON recommendation_impression_views (impression_id, product_id);

CREATE TABLE IF NOT EXISTS recommendation_event_outbox (
    event_id VARCHAR(64) PRIMARY KEY,
    impression_id VARCHAR(64) NOT NULL UNIQUE,
    payload_hash VARCHAR(64) NOT NULL,
    event_payload JSONB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_by VARCHAR(255),
    claim_expires_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_recommendation_event_outbox_pending
    ON recommendation_event_outbox (published_at, next_attempt_at, claim_expires_at);

CREATE TABLE IF NOT EXISTS interaction_idempotency_ledger (
    event_id VARCHAR(64) PRIMARY KEY,
    payload_hash VARCHAR(64) NOT NULL,
    event_payload JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    lease_owner VARCHAR(64),
    lease_expires_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
