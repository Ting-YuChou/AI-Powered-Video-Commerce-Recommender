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

## Restart, rebuild, and JobManager loss

Generations live only in Flink state, so the job must never start from empty
state under an epoch that already wrote Redis. Every such write would be
judged stale, then fail on digest conflicts once generations caught up.

- The submitter (`flink-jobs/interaction-features/bin/submit-interaction-job.sh`)
  restores, in order: `FLINK_RESTORE_SAVEPOINT_PATH`, the newest retained
  checkpoint under `FLINK_CHECKPOINT_DIR` (`file://` only; pass an explicit
  path for remote storage), and only then empty state.
- `./startup.sh restart` (compose down/up) and an image rebuild therefore resume
  from the last completed checkpoint. Events after that checkpoint are replayed
  from Kafka and deduplicated by event ID.
- The Compose JobManager has no HA. If it dies, the job is gone and
  `FlinkInteractionFeatureJobMissing` fires. Re-run the submitter with
  `docker compose --profile flink up -d flink-interaction-features`; it
  restores the newest retained checkpoint.
- `UserFeatureRedisSink` records the epoch owner in
  `flink:feature_state_owner`. A new job without restored state under the
  owner's epoch refuses to start, with an error naming the epoch. The job then
  fails after the bounded restart budget (10 failures in 10 minutes) instead of
  looping.
- To discard state on purpose, bump `FLINK_FEATURE_STATE_EPOCH` and set
  `FLINK_RESTORE_FROM_LATEST_CHECKPOINT=false` for that one submission.
- Redis snapshot conflicts are counted (`FlinkRedisSnapshotConflict`, critical)
  and skipped instead of failing the job. Sustained stale writes raise
  `FlinkRedisSnapshotStaleWrites`.
- Upgrading to this version for the first time: no owner marker exists yet,
  so use the savepoint upgrade above rather than a plain restart.
- Retained checkpoints of previous job IDs are not cleaned up automatically.
  Prune old `/flink/checkpoints/runtime/<job-id>` directories after a healthy
  restore.

`scripts/run_flink_recovery_drill.sh` covers the savepoint restore and a
compose down/up without a savepoint.

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
