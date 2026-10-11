# Flink realtime feature recovery

The production interaction feature job is authoritative for the `official`
Redis namespace. Keep exactly one active job and preserve
`FLINK_FEATURE_STATE_EPOCH` during ordinary savepoint upgrades. Increase the
epoch only when operators intentionally abandon old Flink state and rebuild all
official keys.

## Savepoint upgrade

1. Verify the job: `scripts/flink_realtime_ops.sh verify`.
2. Stop with a retained savepoint:
   `scripts/flink_realtime_ops.sh stop-with-savepoint`.
3. Record the returned path and keep the pre-upgrade jar.
4. Restore without `allowNonRestoredState`:
   `scripts/flink_realtime_ops.sh restore <savepoint-path>`.
5. Verify one running job, recent checkpoints, Redis `ufm:*` generations, and
   zero snapshot conflicts.

The stateful IDs captured before this change are locked in
`flink-jobs/interaction-features/operator-id-manifest.json`. A missing or
incompatible state mapping must fail the restore. Do not bypass it with
`allowNonRestoredState`.

## Late interactions

Interactions beyond the window allowed-lateness bound still update the durable
Postgres interaction row and lifetime user snapshot. They do not reopen closed
5m, 1h, or 24h windows. Each rejected entity/window is emitted to
`late-interaction-events` with a deterministic quarantine ID for reconciliation
or an offline backfill. Never replay this quarantine topic into realtime
windows.

## DLQ replay

The replay command is a dry run unless `--execute` is provided:

```bash
python scripts/replay_kafka_dlq.py \
  --allowed-source-topic user-interactions \
  --audit-output /tmp/dlq-audit.json
```

Review the report, then repeat with `--execute`. The command snapshots partition
end offsets, accepts only allowlisted source topics, requires a stable event ID,
and rejects events older than seven days. Execute mode preserves the source
payload, key, and headers and adds replay identity headers. The Flink eight-day
dedupe state makes repeated replay safe.

Parser failures and corrected transient failures may be replayed. Events without
a stable event ID, events outside the horizon, and payloads that still violate
the schema stay quarantined for manual repair.
