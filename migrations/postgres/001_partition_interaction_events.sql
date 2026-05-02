-- Convert interaction_events to monthly RANGE partitions by occurred_at.
--
-- Use only during a planned maintenance window. The application must be
-- stopped while this runs because the script renames and backfills the table.
-- After applying it, set DATABASE_INTERACTION_EVENTS_PARTITIONED=true so
-- inserts use partition-compatible ON CONFLICT handling.

BEGIN;

LOCK TABLE interaction_events IN ACCESS EXCLUSIVE MODE;

ALTER TABLE interaction_events RENAME TO interaction_events_legacy;

CREATE TABLE interaction_events (
    event_id VARCHAR(64) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    request_id VARCHAR(64),
    user_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    action VARCHAR(64) NOT NULL,
    context JSON NOT NULL DEFAULT '{}'::json,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, occurred_at)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE interaction_events_default
    PARTITION OF interaction_events DEFAULT;

CREATE INDEX ix_interaction_events_occurred_at_desc
    ON interaction_events (occurred_at DESC);
CREATE INDEX ix_interaction_events_action_occurred_at
    ON interaction_events (action, occurred_at DESC);
CREATE INDEX ix_interaction_events_user_id
    ON interaction_events (user_id);
CREATE INDEX ix_interaction_events_product_id
    ON interaction_events (product_id);
CREATE INDEX ix_interaction_events_user_sequence
    ON interaction_events (user_id, occurred_at, event_id);

DO $$
DECLARE
    partition_start DATE := date_trunc('month', now())::date - INTERVAL '3 months';
    partition_end DATE := date_trunc('month', now())::date + INTERVAL '6 months';
    current_start DATE;
    current_end DATE;
    partition_name TEXT;
BEGIN
    current_start := partition_start;
    WHILE current_start < partition_end LOOP
        current_end := current_start + INTERVAL '1 month';
        partition_name := format('interaction_events_%s', to_char(current_start, 'YYYY_MM'));
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF interaction_events FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            current_start,
            current_end
        );
        current_start := current_end;
    END LOOP;
END $$;

INSERT INTO interaction_events (
    event_id,
    schema_version,
    request_id,
    user_id,
    product_id,
    action,
    context,
    occurred_at,
    created_at
)
SELECT
    event_id,
    schema_version,
    request_id,
    user_id,
    product_id,
    action,
    context,
    occurred_at,
    created_at
FROM interaction_events_legacy
ON CONFLICT DO NOTHING;

COMMIT;
