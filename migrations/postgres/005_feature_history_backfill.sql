-- Resumable operational checkpoint for the one-time history backfill.

CREATE TABLE IF NOT EXISTS feature_history_backfill_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    range_start TIMESTAMPTZ NOT NULL,
    range_end TIMESTAMPTZ NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    phase VARCHAR(32) NOT NULL DEFAULT 'catalog',
    cursor_time TIMESTAMPTZ,
    cursor_id VARCHAR(255),
    counts JSON NOT NULL DEFAULT '{}'::json,
    reconciliation JSON NOT NULL DEFAULT '{}'::json,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_feature_history_backfill_active_range
ON feature_history_backfill_runs (range_start, range_end)
WHERE status = 'active';
