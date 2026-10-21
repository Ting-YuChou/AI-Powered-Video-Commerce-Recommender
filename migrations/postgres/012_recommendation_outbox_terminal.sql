-- Terminal state for the served-impression and catalog outboxes.
--
-- A row goes terminal instead of retrying forever when its publish error is
-- not retryable (for example a record over the Kafka size limit), when it has
-- used RECOMMENDATION_OUTBOX_MAX_ATTEMPTS publishes, or (recommendation only)
-- when its insert committed after the serving budget expired and the client
-- never received the impression_id ("not_exposed"). Terminal rows are never
-- claimed, never resolve view slates and are excluded from pending stats.
--
-- SystemStore._ensure_feature_lake_operational_schema applies the same
-- additive ALTERs at startup when DATABASE_AUTO_CREATE_SCHEMA=true.

ALTER TABLE recommendation_event_outbox
    ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS terminal_reason TEXT;

-- The catalog outbox previously retried immediately with no backoff.
ALTER TABLE catalog_feature_outbox
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS terminal_reason TEXT;
