-- Terminal state for the served-impression and catalog outboxes.
--
-- A row goes terminal instead of retrying forever when its publish error is
-- not retryable (for example a record over the Kafka size limit), when it has
-- used RECOMMENDATION_OUTBOX_MAX_ATTEMPTS publishes, or (recommendation only)
-- when its insert committed after the serving budget expired and the client
-- never received the impression_id ("not_exposed"). Terminal rows are never
-- claimed, never resolve view slates and are excluded from pending stats.
--
-- Apply this before deploying (production should run with
-- DATABASE_AUTO_CREATE_SCHEMA=false). SystemStore startup applies the same
-- columns and indexes only when they are missing, but an ALTER on the hot
-- impression outbox during a rolling deploy can stall inserts for up to
-- DATABASE_SCHEMA_LOCK_TIMEOUT_MS.

ALTER TABLE recommendation_event_outbox
    ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS terminal_reason TEXT;

-- The catalog outbox previously retried immediately with no backoff.
ALTER TABLE catalog_feature_outbox
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS terminal_reason TEXT;

-- Claims filter on published_at IS NULL AND terminal_at IS NULL; terminal rows
-- keep published_at NULL, so the older pending indexes no longer narrow the
-- scan. On a large outbox run these one statement at a time outside a
-- transaction (psql autocommit) so CONCURRENTLY can avoid blocking inserts.
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_recommendation_event_outbox_dispatchable
    ON recommendation_event_outbox (next_attempt_at, created_at)
    WHERE published_at IS NULL AND terminal_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_catalog_feature_outbox_dispatchable
    ON catalog_feature_outbox (next_attempt_at, created_at)
    WHERE published_at IS NULL AND terminal_at IS NULL;
