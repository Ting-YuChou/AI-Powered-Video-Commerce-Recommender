# Progress

## 2026-07-12 — Temporal multimodal video ranking and region OCR

- Replaced fixed eight-frame mean-only content features with an adaptive
  8–16-frame contract, trainable temporal encoding, and candidate-conditioned
  visual/OCR attention for a retrained ranking v3 model.
- Added PaddleOCR v5 mobile detection/recognition in an isolated persistent
  content-worker subprocess, plus edit-similarity and polygon-IoU temporal OCR
  tracking that keeps the first occurrence anchor.
- Preserved the legacy pooled embedding for recall/checkpoint compatibility and
  added durable Postgres `content_feature_artifacts` beside the Redis serving
  cache. Ranking runner payload versions 1/2 remain supported alongside v3.
- Key files: `video_commerce/ml/temporal_multimodal.py`,
  `video_commerce/ml/video_ocr.py`, `video_commerce/ml/content_processor.py`,
  `video_commerce/ml/ranking.py`, and
  `migrations/postgres/007_temporal_multimodal_features.sql`.
- Verification: focused Docker backend suite `164 passed`; fresh host suites
  `8 passed` and `66 passed`; dedicated isolated PaddleOCR content-worker image
  built on linux/arm64; application and OCR virtualenv imports passed; Compose
  config and `git diff --check` passed. Helm CLI was unavailable locally.
- Follow-up: apply migration 007, backfill completed durable content jobs, train
  a fresh v3 checkpoint, and gate activation on offline NDCG/CTR/CVR plus
  coverage. Paddle model weights should be pre-cached for production startup.

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
- Follow-up superseded by the greenfield Phase 4 decision below: this checkout
  has no production traffic/artifact to protect, so rollout is PIT-first with
  per-run correctness gates instead of a seven-day shadow window.

## 2026-07-11 — Offline Feature Store phase 4 direct PIT rollout

- Added a leased, deterministic daily `PitDatasetOrchestrator` with Postgres run
  state, `02:00 UTC` cutoffs, deterministic Flink REST job IDs, takeover-safe
  polling/lease renewal, crash-resumable manifest publication,
  attempt-isolated Parquet shards, and terminal empty-run bootstrap behavior
  that never publishes `latest.json`. A new daily invocation first completes
  any expired prior run, then continues its originally requested cutoff.
- Made the primary trainer wait for a missing first pointer, fail closed for a
  malformed manifest, restore/skip already-trained materialization run IDs, and
  stay isolated from Redis/current catalog. Untrained serving now always orders
  by candidate `combined_score`, even when no neural module is initialized.
- Added the tracked Compose `pit-production` preset and dedicated Flink REST
  submitter image, the external-dependency Helm
  `values-pit-production.yaml` preset and daily `concurrencyPolicy: Forbid`
  CronJob, durable trainer claims/idempotent artifact records, a long-lived PIT
  state metrics exporter, rollout alerts/dashboard panels, and greenfield
  bootstrap/retry/rollback operations documentation.
- PIT training renews its Postgres lease while running. Lease loss signals the
  synchronous PyTorch worker cooperatively and waits for that thread to exit
  before releasing the claim. PIT artifact objects are content-addressed;
  checkpoint conflicts load and checksum-verify the durable winner rather than
  overwriting it.
- Greenfield readiness now treats uncommitted Kafka partitions as ready only
  when their broker end offsets are zero, and ignores terminal same-name Flink
  job history while still rejecting multiple active materializers. Flink 1.20
  unknown-job 500 responses are normalized only for its explicit
  `FlinkJobNotFoundException`. PIT REST submissions carry catalog, warehouse,
  S3 endpoint, feature, attribution, and lateness settings as CLI arguments so
  externally managed Flink does not depend on orchestrator container env.
- Key files: `video_commerce/services/pit_dataset_orchestrator/main.py`,
  `migrations/postgres/006_pit_materialization_runs.sql`,
  `flink-jobs/interaction-features/src/main/java/com/videocommerce/flink/PointInTimeFeatureJoinJob.java`,
  `deploy/compose/pit-production.conf`,
  `charts/video-commerce/values-pit-production.yaml`, and
  `docs/operations/offline-feature-store.md`.
- Verification: final pre-fallback Docker backend `433 passed, 4 skipped`; the
  subsequent micro-batch fallback regression suite passed `5` focused tests.
  Flink Maven passed all 25 tests. Default and PIT Compose config, Helm lint,
  PIT production render, strict kubeconform (36 resources), Prometheus rule
  tests, Python compile, and diff checks passed. A clean Feature Lake runtime
  reached healthy MinIO, Iceberg, Flink, state-exporter, trainer, and a
  restart-safe readiness exit. Deterministic run `pit-20260711` ended in
  `waiting_for_eligible_rows` with zero rows and no training prefix or
  `latest.json`. Direct serving verification proved the untrained micro-batch
  path orders `combined_score=0.9` before `0.1`, reports
  `fallback_untrained`, and performs no neural forward. A public hot load
  completed 3000/3000 HTTP 200 requests; it had no candidates and therefore is
  not treated as fallback model performance evidence.
- Follow-up: run matched fallback/trained serving load gates after the first
  legitimate 168-hour mature PIT manifest and artifact exist. The attempted
  1000-request internal fallback load was blocked by local execution quota
  after the direct runtime assertion succeeded. Host-wide Black remains noisy
  on pre-existing formatting; changed-line compile and diff checks passed.

## 2026-07-14 — Three-modal temporal video ranker V4

- Added timestamped Qwen3 forced alignment with degraded transcript-preserving
  fallback; frozen pinned multilingual E5 embeddings for OCR, ASR, and catalog
  text; real frame timestamps; and bounded 16/32/64 visual/OCR/ASR sequences.
- Published canonical immutable `temporal_multimodal_v2` content artifacts and
  propagated URI/SHA/schema references into PIT examples. The PIT reader now
  verifies referenced bytes and fails closed on checksum or schema mismatch.
- Added three temporal encoders, candidate-conditioned cross-attention,
  presence-aware modality gating, 10% OCR/ASR dropout, V3 warm-start, first
  epoch base freeze, differential learning rates, and the
  `ranking_v4_00_temporal_trimodal` checkpoint schema.
- Added bounded payload V4 support across recommendation, ranking service,
  coordinator, runner, and local fallback; checksum-locked candidate NPZ
  sidecars with atomic activation; V1-V3 compatibility; shadow-by-default
  rollout and V3 checkpoint rollback lineage.
- Key files: `asr_service/api.py`, `video_commerce/ml/content_processor.py`,
  `video_commerce/ml/content_artifacts.py`,
  `video_commerce/ml/temporal_multimodal.py`, `video_commerce/ml/ranking.py`,
  `video_commerce/ml/ranking_training.py`, and
  `video_commerce/ml/candidate_embedding_sidecar.py`.
- Verification: 107 focused host tests and the full Docker backend regression
  passed (458 passed, 4 skipped). Python compile, Black, diff checks, and
  `docker compose config -q` passed. The ASR and content-worker production
  images built successfully, and the ASR image imported `qwen-asr` with the
  pinned forced-aligner model. Helm/kubeconform could not run because local
  binaries are unavailable and the sandbox rejected mounting the private repo
  into the Helm container. GPU 300-second ASR profiling, v3/v4 candidate
  benchmarks, and offline NDCG/AUC/GMV promotion evaluation require production
  hardware/data and remain rollout gates. No performance improvement is
  claimed.
