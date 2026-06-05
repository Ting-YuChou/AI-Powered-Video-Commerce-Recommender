-- Add a partial index for positive-action user sequence training reads.
--
-- Safe to run on existing non-partitioned or partitioned interaction_events
-- tables. On partitioned tables, Postgres propagates the partitioned index to
-- child partitions.

CREATE INDEX IF NOT EXISTS ix_interaction_events_positive_user_sequence
    ON interaction_events (user_id, occurred_at DESC, event_id DESC)
    WHERE action IN ('view', 'click', 'add_to_cart', 'purchase');
