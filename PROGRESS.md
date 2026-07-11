# Progress

## 2026-07-09 — Ranking PIT feature-store foundation

- Added a shared versioned ranking feature contract and deterministic `as_of_ts`
  extraction for training; PIT rows now override current online user/item maps.
- Governed public interaction timestamps with `event_time`, immutable
  `server_received_at`, 30-day historical and 5-minute future bounds, while
  retaining `timestamp` as a backwards-compatible alias. Flink prefers the new
  event time.
- Added feature-lake rollout configuration, a manifest-validated PIT dataset
  reader, and a trainer path that fails closed in `FEATURE_LAKE_TRAINING_SOURCE=pit`
  rather than falling back to Redis/Postgres current-state features.
- Upgraded the Flink project/runtime configuration to 1.20.1, added the Iceberg
  1.11 runtime, and added a batch PIT SQL job with feature event/availability
  predicates and the 7-day attribution window. Helm now leaves the legacy
  Python feature worker disabled by default under the Flink-authoritative path.
- Key files: `video_commerce/common/event_time.py`,
  `video_commerce/ml/point_in_time.py`,
  `video_commerce/ml/pit_training_dataset.py`,
  `video_commerce/services/model_trainer/main.py`,
  `flink-jobs/interaction-features/`.
- Verification: Docker backend tests for event time, config, PIT contracts,
  trainer routing, ranking LTR/features; isolated Maven container `mvn test`;
  `docker compose config -q`.
- Follow-up: materialize Kafka interaction/catalog records into the declared
  Iceberg history tables and publish the completed snapshot as the manifest
  export consumed by the trainer; then run an end-to-end MinIO/REST catalog
  integration test before enabling `FEATURE_LAKE_TRAINING_SOURCE=pit`.

## 2026-07-10 — Offline Feature Store phase 2 materialization and PIT export

- Added deterministic Python/Java history contracts, catalog activation/outbox
  publishing, shared user/window snapshots, recommendation observations,
  resumable backfill reconciliation, and an independent Flink Iceberg
  materializer for five append-only history tables.
- Added fail-closed PIT joins and immutable run IDs, final bundle hash
  recomputation, Parquet manifests pinned to an Iceberg snapshot, trainer
  validation, replay namespaces, a checkpoint/zero-lag readiness gate, shadow
  gates, metrics, alerts, Compose/Helm configuration, and the operations guide.
- Runtime validation exposed and fixed a reserved DLQ SQL column, Kafka DLQ
  transaction timeout, missing topic initialization, Python/Jackson float
  canonicalization, Iceberg `DOUBLE` cutoff predicates, TaskManager metaspace,
  the missing Flink Parquet format, S3A client endpoint propagation, extensionless
  Flink shard discovery, and atomic S3 create-only header compatibility.
- A fresh isolated v6 replay wrote one row to each of the five history tables;
  a duplicated catalog event remained one distinct `source_event_id`. PIT v6f
  produced one click-labelled row with no quarantine rows, exported an immutable
  Parquet shard, published a snapshot-pinned completed manifest with parity 1.0,
  and loaded successfully through the trainer reader. Republishing the same run
  failed with S3 `PreconditionFailed`, as required.
- Key files: `video_commerce/common/feature_history_contracts.py`,
  `video_commerce/data_plane/system_store.py`,
  `video_commerce/ml/pit_manifest.py`,
  `flink-jobs/interaction-features/src/main/java/com/videocommerce/flink/FeatureHistoryMaterializerJob.java`,
  `flink-jobs/interaction-features/src/main/java/com/videocommerce/flink/PointInTimeFeatureJoinJob.java`,
  `docs/operations/offline-feature-store.md`.
- Verification: backend `378 passed, 4 skipped`; complete Maven tests passed;
  Compose config, Helm lint/template, kubeconform strict (33 valid), and
  Prometheus rule tests passed. Fresh Kafka -> Flink -> Iceberg -> PIT ->
  Parquet -> manifest -> trainer runtime smoke and readiness gate passed.
- Follow-up: keep `FEATURE_LAKE_TRAINING_SOURCE=legacy` until the production-like
  seven-day shadow gates pass. Remaining rollout work is operational scheduling,
  real p99 lag/parity evidence, schema-evolution rehearsal for existing Iceberg
  tables, and catalog activation cutover before training switches to PIT.

## 2026-07-10 — Offline Feature Store phase 3 typed PIT training

- Replaced PIT dictionary heuristics with typed `FeatureBundle`,
  `AttributionFacts`, `RankingTrainingExample`, and `TrainingLabels` contracts.
  The manifest reader now pins Iceberg table/snapshot IDs, validates all typed
  row fields and versions, and returns examples rather than raw dictionaries.
- Unified online batching and offline tensor construction behind the versioned
  `RankingFeatureAssembler`; PIT features use bundle `as_of_ts` only and reject
  unsupported versions, shape mismatches, and non-finite values. Added the
  `ranking_labels_v1` builder with attribution-only CTR/CVR/CTCVR, masks,
  relevance, and actual-feedback purchase value semantics.
- Isolated current Redis/catalog access in `LegacyTrainingDatasetAdapter`.
  Primary PIT trainer initialization no longer loads Redis features, vector
  catalog state, or recommendation models. Added non-activating
  `ranking_model_pit_shadow` artifacts, typed rollout metrics/alerts, and
  feature/label/assembler/manifest lineage in checkpoints and artifact metadata.
- Added explicit Iceberg additive schema evolution and runtime-resolved insert
  column ordering so an existing Phase 2 table can safely append Phase 3 label
  columns. Existing v6 table schema evolved from schema ID 0 (18 columns) to
  schema ID 4 (22 columns); run `smoke-pit-20260710-phase3c` committed one typed
  row with zero quarantine rows at snapshot `5595377083784054562`.
- Key files: `video_commerce/ml/ranking_features.py`,
  `video_commerce/ml/ranking_training.py`,
  `video_commerce/ml/pit_training_dataset.py`,
  `video_commerce/ml/legacy_training_adapter.py`,
  `video_commerce/services/model_trainer/main.py`, and
  `flink-jobs/interaction-features/src/main/java/com/videocommerce/flink/PointInTimeFeatureJoinJob.java`.
- Verification: Docker backend `396 passed, 5 skipped`; Flink Maven 23 tests
  passed; Compose config, Helm lint, strict kubeconform (33 resources),
  Prometheus rules, Grafana JSON, Python compile/diff checks, and Black checks
  passed. Existing Iceberg table migration, PIT export, completed manifest,
  typed reader, shared assembler/labels/trainer, and checkpoint smoke passed.
- Follow-up: keep `FEATURE_LAKE_TRAINING_SOURCE=legacy` and enable PIT shadow
  only. A real seven-day shadow window, online load p95/throughput comparison,
  and production parity/coverage/lag evidence remain rollout gates rather than
  one-time local test claims.
