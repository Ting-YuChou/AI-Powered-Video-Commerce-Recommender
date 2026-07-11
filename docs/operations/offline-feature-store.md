# Offline Feature Store Operations

The offline store is Kafka to Flink to Iceberg on S3-compatible object storage.
Redis remains the online serving store. Postgres remains operational state and
holds only catalog outbox, activation, and one-time backfill checkpoints.

## Local feature-lake stack

Start the streaming stack without changing the trainer source:

```bash
FEATURE_LAKE_TRAINING_SOURCE=legacy \
docker compose --profile feature-lake up -d --build
```

This starts MinIO, the Iceberg REST fixture, external-style Flink cluster, the
independent history materializer, and catalog outbox publisher. The online
recommendation path performs no Iceberg I/O.

`feature-history-snapshots` is reserved for immutable Flink user/window
snapshots. The legacy Python/content-worker `feature-updates` topic remains
operational and is deliberately excluded from history materialization.

Run the one-time Postgres backfill with a fixed immutable range. Reusing the
same run ID resumes the `(time, id)` cursor. A second active run for the same
range is rejected.

```bash
docker compose --profile feature-lake run --rm catalog-event-publisher \
  python -m video_commerce.services.feature_history_backfill.main \
  --run-id initial-2026-07 --range-start 0 --range-end 1783555200
```

Set `FEATURE_HISTORY_INCLUDE_BACKFILL_TOPICS=true` only while reconciling that
run. Historical impressions without a complete `ranking_ltr_v1` bundle are
sent to `feature-history-backfill-quarantine`; the command never invents their
missing fields.

After the backfill topics are fully consumed, reconcile the immutable Kafka
accepted count against Iceberg accepted rows plus DLQ rows. The run is marked
complete only when the counts are exact and duplicate source IDs are zero:

```bash
docker compose --profile feature-lake run --rm catalog-event-publisher \
  python -m video_commerce.services.feature_history_backfill.main \
  --run-id initial-2026-07 --reconcile
```

The command captures bounded Kafka end offsets and scans the five Iceberg
tables by `backfill_run_id`; it persists the exact snapshot IDs and refuses to
complete unless distinct Kafka source IDs equal Iceberg plus DLQ IDs with zero
Iceberg duplicates.

Flink 1.20.1 runs on Java 17 in this profile because Iceberg 1.11 artifacts are
Java 17 bytecode. User and item history use a deterministic physical
`entity_bucket` column because Flink DDL does not support Iceberg hidden bucket
partition transforms.

## Replay

Never reset production materializer offsets. Set `FEATURE_LAKE_REPLAY_RUN_ID`
for a full replay; the job derives an isolated namespace such as
`video_commerce_rebuild_run_42` and a separate consumer group. Validate row
counts, duplicate source IDs, payload hashes, and PIT leakage there before a
REST Catalog rename/promote operation.

## PIT export and manifest

The batch job first commits `ranking_training_pit` to Iceberg and then emits
immutable Parquet shards. Obtain the exact committed Iceberg snapshot ID and
run:

```bash
FEATURE_LAKE_MATERIALIZATION_RUN_ID=shadow-2026-07-09 \
FEATURE_LAKE_PIT_ICEBERG_SNAPSHOT_ID=<exact-snapshot-id> \
docker compose --profile feature-lake-pit up --abort-on-container-exit \
  --exit-code-from pit-manifest-publisher
```

The manifest publisher verifies every shard schema, row count, byte size, and
SHA-256 before writing `manifest.json`; `latest.json` is written last. The
trainer resolves `latest.json` once per training run and fails closed on any
manifest, shard, schema, hash, or feature-definition mismatch.

## Seven-day cutover and rollback

Keep `FEATURE_LAKE_TRAINING_SOURCE=legacy` until seven consecutive UTC daily
reports pass all gates: zero leakage and duplicates, complete reconciliation,
99.9% bundle parity at `1e-6`, lag/outbox thresholds, coverage within 0.1
percentage point, and 100% shadow training/artifact persistence.

Then set:

```text
FEATURE_LAKE_TRAINING_SOURCE=pit
FEATURE_LAKE_RANKING_PIT_DATASET_URI=s3://video-commerce-features/training/ranking-pit/latest.json
```

Rollback changes only `FEATURE_LAKE_TRAINING_SOURCE` to `legacy`. Publisher,
materializer, or batch jobs may be paused independently and do not affect
online recommendation serving.

Kubernetes remains application-only. The Helm chart can deploy
`catalogEventPublisher`, but Postgres, Kafka, S3, Iceberg REST, and the Flink
cluster/jobs must be externally managed.
