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

New Phase 3 manifests must also declare `label_definition_version` as
`ranking_labels_v1`. Their shards contain finalized attribution facts rather
than precomputed model targets. The typed reader validates every observation,
impression, user/item snapshot, candidate score, version, and hash before it
constructs a `RankingTrainingExample`. Actual purchase value comes only from
feedback inside the attribution window; missing value remains null and produces
`value_mask=0`.

The PIT trainer path is deliberately isolated from current Redis features and
the current catalog. It executes the same versioned feature assembler used by
online ranking, then the versioned label builder. A malformed row, unsupported
feature/label version, non-finalized attribution, or NaN/Inf fails the entire
PIT training run instead of falling back to legacy data.

## Typed PIT shadow training

To train a non-activating PIT artifact while legacy remains primary:

```text
FEATURE_LAKE_ENABLED=true
FEATURE_LAKE_TRAINING_SOURCE=legacy
FEATURE_LAKE_PIT_SHADOW_ENABLED=true
FEATURE_LAKE_RANKING_PIT_DATASET_URI=s3://video-commerce-features/training/ranking-pit/latest.json
```

Shadow checkpoints are persisted under the separate
`ranking_model_pit_shadow` artifact name with `activation_allowed=false`.
Serving never watches that namespace. The trainer records the pinned manifest,
materialization run, Iceberg snapshot, schema hash, feature/label/assembler
versions, input and quarantine counts, and value-mask coverage in artifact
metadata.

## Greenfield PIT-first rollout

This environment has no production traffic or artifact to preserve, so the
production-like Compose preset uses PIT as the primary source immediately. It
does not shorten the 168-hour attribution window or create an empty manifest.

```bash
docker compose --env-file deploy/compose/pit-production.conf \
  --profile feature-lake --profile pit-production up -d --build
```

The first healthy state is normally `waiting_for_eligible_rows` in
`pit_materialization_runs` and `waiting_for_pit_manifest` in trainer metrics.
That state is expected until observations are older than the attribution window
plus allowed lateness. `latest.json` remains absent. The trainer never reads
current Redis/catalog state and does not fall back to legacy. Until a valid PIT
artifact is activated, ranking sorts candidates by their existing
`combined_score`; it does not execute a randomly initialized neural model.

Every daily cutoff is `02:00:00 UTC` with deterministic run ID
`pit-YYYYMMDD`. Retry a particular UTC run without changing its identity:

```bash
docker compose --env-file deploy/compose/pit-production.conf \
  --profile feature-lake --profile pit-production run --rm \
  pit-dataset-orchestrator --once --date 2026-07-11
```

Waiting runs are terminal for that daily cutoff; the next daily run reevaluates
newly eligible data. Failed runs are retryable. A run that already reached
the manifest phase resumes there without appending a second PIT Iceberg run.
An identical completed manifest may be republished to repair `latest.json`, but
different content under the same run ID fails closed.

Inspect durable orchestration and artifact lineage:

```bash
docker compose exec -T postgres psql -U video_commerce -d video_commerce \
  -c "select run_id,status,phase,attempts,snapshot_id,manifest_uri,row_count,last_error from pit_materialization_runs order by cutoff_at desc limit 10"

docker compose exec -T postgres psql -U video_commerce -d video_commerce \
  -c "select model_name,model_version,payload->>'feature_lake_materialization_run_id' as pit_run from model_checkpoints order by created_at desc limit 10"
```

The Helm preset is application-only and deliberately fails rendering until
external Kafka, S3, Iceberg REST, and Flink endpoints are supplied:

```bash
helm upgrade --install video-commerce charts/video-commerce \
  -f charts/video-commerce/values-pit-production.yaml \
  --set external.kafka.bootstrapServers=kafka.example:9092 \
  --set external.objectStorage.endpointUrl=https://s3.example \
  --set external.objectStorage.bucket=video-commerce-features \
  --set external.featureLake.catalogUri=https://iceberg.example \
  --set external.featureLake.warehouseUri=s3://video-commerce-features/warehouse \
  --set external.featureLake.flinkJobmanager=flink.example:8081
```

The CronJob runs daily at `02:00 UTC` with `concurrencyPolicy: Forbid` and
submits a deterministic Job ID through the Flink REST API. The long-lived
`pit-state-exporter` exposes Postgres-backed last-success, waiting, running, and
expired-lease state after CronJob pods exit. REST-submitted application setup
runs on the external JobManager, so both JobManager and TaskManager images must
provide the cluster-level runtime used by the Compose feature image:

- Flink 1.20.1 on Java 17.
- Hadoop client API/runtime 3.4.2 in `/opt/flink/lib`.
- The Flink S3 filesystem plugin enabled on every node.
- Network access and credentials for the configured Iceberg REST catalog and
  S3-compatible warehouse.

The application JAR carries Iceberg 1.11 and the PIT code, while catalog,
warehouse, S3 endpoint, feature version, attribution window, and allowed
lateness are passed explicitly through the REST `programArgsList`. Credentials
remain cluster-managed and are never sent as job arguments.

Activation still requires each individual run to pass manifest/hash/schema,
PIT leakage, typed assembler, label reconciliation, and finite-tensor gates.
The seven-day shadow gate is not used for this greenfield rollout.

Emergency rollback changes only `FEATURE_LAKE_TRAINING_SOURCE=legacy` and
restarts the trainer. Kafka-to-Iceberg materialization may continue; serving
never reads Iceberg directly.
