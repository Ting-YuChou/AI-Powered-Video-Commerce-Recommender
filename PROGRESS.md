# Progress

## 2026-10-08 — Automated deployment gates, Flink closed loop, and outbox A/B harness

- Expanded PR CI with Node 20 frontend tests/lint/build, Java 11 Flink Maven
  tests, pinned Helm/kubeconform/promtool deployment validation, and an isolated
  `flink-closed-loop` check with failure artifacts and project-scoped cleanup.
- Added a real interaction-to-serving smoke: one stable click becomes one
  Postgres row and one official Redis feature/sequence update, an identical
  retry remains exactly once, and the next durable impression snapshot proves
  recommendation serving consumed `total_interactions=1`.
- Fixed the CI-discovered ranking payload failure for Flink-derived nested maps:
  non-string JSON object keys are normalized at the ranking HTTP boundary, while
  string-conversion collisions fail explicitly instead of overwriting data.
- Extended the HTTP baseline with run isolation, excluded warm-up, successful
  latency/QPS, cache, transport/5xx, and durable tracking metrics. Added the
  three-pair hot/unique served-impression outbox A/B runner, exact published-row
  reconciliation, resource/artifact lineage, and all specified acceptance gates.
- Key files: `.github/workflows/ci.yml`,
  `scripts/run_flink_closed_loop_smoke.sh`,
  `tests/integration/test_flink_closed_loop.py`,
  `scripts/loadtest_api_baseline.py`, and
  `scripts/run_recommendation_outbox_ab.py`.
- Verification: isolated Compose/Flink smoke passed again after the ranking
  payload fix (`1 passed in 3.43s`) and removed its fresh project volumes;
  the complete ranking optimization suite passed (`79 passed`), and the
  baseline/A/B and vector-regression tests passed (`26 passed`); frontend 14 tests,
  lint, and build passed; Flink Maven 28 tests passed; Helm lint, strict
  kubeconform `36/36`, Prometheus rule tests, Compose config, shell/Python syntax,
  and diff checks passed.
- Known gaps: the full three-pair 3,000-request hot/unique A/B run has not been
  executed, so latency/QPS and outbox-cost gates remain unverified and no
  performance-regression claim is made. The smoke disables the known-user
  negative snapshot cache because its production refresh interval can hide a
  newly created user's fresh Flink features. Checkpoint restore, artifact rolling
  activation, DLQ replay, and Iceberg cleanup drills remain follow-up work.

## 2026-10-08 — Content upload outbox and worker processing leases

- Added migration `009_content_task_reliability.sql` for durable content-task
  publication, pipeline-versioned job state, and owner-fenced processing runs.
  Uploads now persist object plus DB/outbox state before publishing, return
  `queued` only after Kafka acknowledgement, and return a traceable
  `CONTENT_ENQUEUE_PENDING` 503 while background retries continue.
- Added the horizontally safe `content-task-publisher` to Compose and Helm. It
  uses leased outbox claims, stable event IDs, exponential retry, missing-object
  terminalization, managed-prefix orphan cleanup after a grace period, and
  seven-day published-row retention.
- Content workers now prevent duplicate model computation per content and
  pipeline version, renew and fence processing leases, commit durable feature
  artifacts before Redis/vector projection, and repair projections from an
  already-completed artifact after redelivery.
- Added content outbox, orphan, duplicate, lease-conflict, lease-loss, and
  projection-repair metrics and alerts. Updated the event/artifact reliability
  guide and deploy/rollback sequence.
- Key files: `migrations/postgres/009_content_task_reliability.sql`,
  `video_commerce/data_plane/system_store.py`,
  `video_commerce/services/content_task_publisher/main.py`,
  `video_commerce/services/content_worker/video_processor.py`, and
  `video_commerce/services/gateway/api.py`.
- Verification: Docker backend suite `644 passed, 10 skipped`; Postgres-backed
  outbox/lease and worker suite `10 passed`; scoped Black, Python compile,
  Compose config, Prometheus rule tests, Helm lint, strict kubeconform `36/36`,
  and production image build passed. A fresh `vc-phase-a-smoke` project proved
  real Kafka/Postgres publication and recovery: the ack-before-mark event was
  observed twice with one stable ID, the worker model ran once across a
  redelivery, projections ran twice, and durable repair resolved `completed`.
- Follow-up: hot/unique load baselines are outside this phase and remain unrun;
  no performance-regression claim is made.

## 2026-10-08 — Flink, event lineage, artifacts, and scoring reliability

- Made the default startup path Flink-aware with idempotent named-job submission,
  exact-one-job health checks, and a missing/duplicate-job alert. Content worker
  failures now update status and rethrow so Kafka retry/DLQ owns offset commits.
- Added fail-closed vector bootstrap modes and checksum/dimension/count manifests;
  local artifacts publish immutable generations through one atomic active pointer,
  while configured object-storage visual-index bundles sync and activate directly.
  Sample data is created only by the explicit demo bootstrap service. Production
  Helm values require verified vector and ranking artifacts.
- Added UUID interaction idempotency through a leased Postgres ledger, durable
  served-impression outbox dispatch, viewed-impression API/frontend tracking,
  Flink viewed materialization, orphan isolation, reconciliation metrics, and
  viewed-only negative sampling. Migration: `008_recommendation_event_reliability.sql`.
- Follow-up two-axis review added owner-token-safe Postgres leases, Postgres
  readiness, original content-error preservation, durable outbox retry recognition,
  durable-slate validation for viewed events, attribution-window enforcement, and
  viewed-history gating in the authoritative Flink PIT materialization.
- Added the versioned `business-value-v1` Torch/NumPy score policy across training,
  NDCG evaluation, model selection, and serving post-processing. Exact-version
  Triton materialization now enforces the same policy and normalization metadata. Impression
  snapshots and artifact metadata retain raw score, serving score, CTCVR,
  predicted value, policy version, and deterministic product-ID tie breaking.
- Key files: `startup.sh`, `video_commerce/data_plane/system_store.py`,
  `video_commerce/services/interaction_ingest/api.py`,
  `video_commerce/services/recommendation/api.py`,
  `video_commerce/ml/ranking_score.py`, `video_commerce/ml/vector_search.py`,
  `flink-jobs/interaction-features/`, and
  `docs/operations/event-and-artifact-reliability.md`.
- Verification: offline backend Docker suite 629 passed/9 skipped; Flink Maven suite 28
  passed; frontend lint, 14 tests, and production build passed; Compose config,
  shell/Python syntax, Helm lint, and kubeconform 33/33 passed.
- Known gaps: the live Compose integration run rebuilt all affected images but
  stopped before tests because the existing Kafka/ZooKeeper volumes have a stale
  cluster-ID mismatch (`rGFua...` versus `dM0r...`). Containers were removed and
  volumes preserved. Hot/unique load baselines therefore remain unrun; no latency
  or production-readiness claim is made from the offline checks.

## 2026-10-07 — Source-grounded project strengthening review

- Reviewed split-service data capture, Kafka/Flink features, retrieval/ranking,
  PIT training, artifacts, frontend, CI, and deployment defaults at `5695367`.
  Added a prioritized report with 23 findings, source locations, model ablations,
  acceptance criteria, and a staged implementation roadmap; runtime is unchanged.
- Confirmed Compose default/PIT profile gaps, content-worker exceptions returning
  handled to the Kafka retry wrapper, HTTP retry event-ID instability, synthetic
  catalog bootstrap, serving/evaluation score mismatch, and scaling opportunities.
- Key files: `docs/reviews/2026-10-07-project-strengthening-review.md`, `PROGRESS.md`.
- Verification: Compose config and filtered profile rendering; 15 PIT/contract
  tests plus 46 synchronous ranking/DIN tests passed (16 async deselected); two
  isolated source-method probes; frontend lint/build passed. Frontend tests first
  failed under Node 26 Web Storage, then 14 passed with
  `NODE_OPTIONS=--no-experimental-webstorage`; no source fix was applied.
- Gaps: Docker daemon unavailable; broader host tests need pytest-asyncio,
  cv2/faiss/redis. No live Kafka/Flink, full backend/integration suite, model-quality
  training, Helm validation, or new capacity benchmark. Findings remain open;
  this task produced the review, not the proposed implementation.

## 2026-08-12 — Opt-in ONNX Runtime and Triton ranking backend

- Added an offline opset-17 ONNX export contract for verified DCN and DCN+DIN
  ranking artifacts. Export validates fixed tensor contracts, source/checkpoint
  and DIN-sidecar lineage, ONNX checker output, finite results, exact top-k, and
  PyTorch/ONNX Runtime parity for batch sizes 1/20/64 plus empty, sparse, and
  full DIN histories. Exact business-version lookup and checksum-derived Triton
  numeric versions prevent ambiguous `latest` activation.
- Added a fail-closed ranking adapter that keeps feature assembly, DIN tensors,
  value calibration, and top-k in `ranking-service`, while sending one
  candidate matrix per async gRPC call to Triton's bounded dynamic batcher.
  Adapter inflight, preprocessing, inference, postprocessing, readiness, request
  ID/deadline propagation, 429 overload, 503 availability, and no sent-request
  retry/fallback semantics are covered by tests. Legacy coordinator/runner
  remains the default rollback backend; trimodal models remain legacy-only.
- Added opt-in Compose and Helm Triton 26.07 deployment/materialization,
  immutable exact-version repositories, idempotent same-artifact startup,
  legacy-volume ownership migration, strict readiness, ServiceMonitor,
  NetworkPolicy, metrics/alerts, an architecture decision record, and an
  interleaved matched A/B load runner. The existing full-path load harness can
  now publish the synthetic verified checkpoint and its ONNX artifact together.
- Key files: `video_commerce/ml/ranking_onnx.py`,
  `video_commerce/ranking_runtime/ranking_triton.py`,
  `video_commerce/services/ranking_service/proxy_asgi.py`,
  `charts/video-commerce/templates/ranking-triton.yaml`, and
  `docs/architecture/decisions/001-ranking-triton-adapter.md`.
- Verification: scoped Docker regression suite `251 passed`; real base and DIN
  export/parity executed inside that suite; Compose config; scoped Black; diff
  checks; default/Triton Helm lint; strict kubeconform `36 valid`; Prometheus
  check/test `43 rules`; production ranking-service/materializer image builds.
  A fresh synthetic artifact was trained, verified, exported, and persisted in
  an isolated Compose project. After reclaiming only dangling images and unused
  build cache, Triton 26.07 pulled successfully. Live startup exposed missing
  protobuf-list commas in generated `config.pbtxt`; a failing regression was
  added before fixing the generator. Exact-model readiness, HTTP 200 ranking,
  and Triton counters then proved one request executed 20 inference rows.
- Read-only review against base `4949962` found no P0 and one P1: legal
  candidate sets above Triton's 256-row limit would fail as one oversized RPC.
  The adapter now sends ordered, non-retried chunks under one overall deadline
  and concatenates all model outputs before scoring. Its 300-row regression and
  the focused adapter/proxy/load suite passed `26` tests; the same reviewer
  confirmed the P1 resolved with no new P0/P1 findings.
- Directional OrbStack ARM short A/B, using the same 20-candidate payload,
  synthetic checkpoint, one adapter process, and one legacy runner: at 500
  offered QPS, Triton delivered 499.11 successful QPS, 0.16% rejection, zero
  5xx, and 8.82 ms successful p95; legacy delivered 459.51 successful QPS,
  7.19% rejection, zero 5xx, and 286.21 ms p95. This is about 8.6% successful
  QPS improvement and 96.9% lower p95, below the formal 10% promotion gate. At
  1000 offered QPS Triton delivered 745.07 successful QPS with 25.33% fast
  rejection, zero 5xx, and 168.69 ms successful p95, identifying local
  saturation rather than a passing case. Raw JSON stayed outside the repo.
- Staged Triton tuning at 1000 offered QPS selected two CPU instances, 4 ms
  queue delay, and ORT intra/inter-op threads of one. Relative to one instance,
  two instances raised successful QPS from 745.07 to 904.76 and reduced p95
  from 168.69 ms to 111.12 ms. With two instances, 1/2/4 ms queue delays
  delivered 900.78/904.76/918.80 successful QPS and
  120.94/111.12/101.13 ms p95 respectively. Raising intra-op threads to two
  regressed to 902.54 successful QPS and 126.01 ms p95. The winner still had
  8.08% fast rejection at 1000 QPS; confirmation runs at 900 and 800 QPS had
  2.04% and 1.48% rejection, so none passed the <0.5% gate. All cases had zero
  5xx. During 900 QPS, one adapter process used 91.69% CPU and 306.6 MiB while
  Triton used 14.14% CPU and 38.4 MiB, identifying adapter feature/pre-post work
  as the next local bottleneck. Winner counters averaged about 108.5 rows per
  execution, 3.02 ms queue time per request, and 1.64 ms compute per execution.
- Follow-up: run the scripted three-repetition matched legacy/Triton A/B and
  30-minute winner soak on a dedicated Linux CPU host. Base and DIN must each
  improve maximum sustained 2xx QPS by at least 10% without exceeding the p95,
  5xx, parity, or fallback gates before staging enablement.

## 2026-08-10 — Full recommendation-path capacity harness and local baseline

- Added a deterministic catalog and synthetic trained/verified ranking
  checkpoint generator for capacity-only tests, plus a k6 scenario that drives
  Caddy through recommendation serving, candidate-cache lookup, coordinator
  batching, ranking runners, and actual Torch model forward. Added path,
  fallback, batch-fill, model-forward, 429/5xx, dropped-iteration, and
  successful-latency evidence to the exported results.
- Ran a fresh isolated OrbStack Compose stack with one, two, and four runners.
  The complete path did not pass the 250-QPS gate, so 500/750/1,000 QPS were
  deliberately not run. Two runners were directionally best, while four
  runners increased CPU contention; increasing max queue wait to 500 ms formed
  larger batches but produced p95 762 ms, p99 1.4 s, 0.57% 5xx, and dropped
  work. The low-rate 300-user prewarm completed without errors and confirmed
  real model-forward execution with zero untrained fallback.
- Key files: `loadtest/k6/recommendations.js`,
  `video_commerce/loadtest_support.py`,
  `scripts/prepare_recommendation_loadtest.py`,
  `scripts/summarize_recommendation_loadtest.py`, and
  `loadtest/results/recommendation-path-20260810/README.md`.
- Verification: focused host suite `4 passed`; Docker checkpoint/artifact/
  feature-contract suite `31 passed`; Python compilation; Compose config; k6
  script inspection; and `git diff --check`. Full result JSON, checkpoint
  lineage, runner matrix, resource snapshot, limitations, and the next staged
  test gate are recorded with the artifacts. The isolated Compose project and
  its disposable volumes were removed and OrbStack was stopped after testing.
- Follow-up: rerun on a dedicated Linux target with the load generator on a
  separate host and fixed resource limits. Pass 250 QPS twice before ramping to
  500/750/1,000 QPS; use a genuinely trained artifact for production
  acceptance and ranking-quality claims.

## 2026-08-08 — Ranking high-concurrency safety and capacity control

- Made ranking production serving fail closed unless the loaded model is
  trained, Torch-ready, and backed by checksum-verified artifact metadata whose
  feature schema matches the active contract. Local/test fallback is now an
  explicit degraded mode, and runner/coordinator readiness exposes model
  lineage plus the minimum healthy-runner requirement.
- Added capacity-aware coordinator workers and bounded internal batch queues,
  queue/slot EWMA admission control, immediate zero-capacity rejection, and a
  stable HTTP 429 overload contract with `Retry-After` propagated through the
  ranking proxy, recommendation service, and gateway. Added runner drain,
  post-send no-retry protection, DNS-safe connection retirement, and rolling
  deployment controls.
- Changed the baseline to 64 max / 32 target requests, 16 ms batching, a 2,048
  request queue, 100 ms max queue wait, four runners with one batch each, and
  two ranking HTTP workers. Added runner/capacity/admission metrics, alerts,
  production-safe Helm probes/PDB/HPA stabilization, a 20-candidate k6
  acceptance scenario, and an evidence checklist in the load-test runbook.
- Key files: `video_commerce/ml/ranking.py`,
  `video_commerce/ranking_runtime/ranking_batcher.py`,
  `video_commerce/ranking_runtime/ranking_runner_client.py`,
  `video_commerce/services/ranking_runner/main.py`,
  `video_commerce/services/ranking_coordinator/main.py`, and
  `charts/video-commerce/`.
- Verification: Docker backend suite `551 passed, 8 skipped`; ranking
  fail-closed/capacity focused suite `54 passed`; `docker compose config -q`; Helm
  lint/template; kubeconform strict `33 valid`; Prometheus rule tests; k6
  script inspection; Black and `git diff --check`. PR CI later exposed three
  legacy shape/fallback tests whose Compose-provided local fallback setting had
  hidden their test-only dependency; those tests now opt into unverified or
  untrained inference explicitly while production defaults remain fail closed.
- Follow-up/blocker: the repository contains no quality-trained production
  checkpoint. A later synthetic-checkpoint matrix exercised real forward passes
  for capacity diagnostics only; the 1,500/2,000/2,500 QPS production soak,
  overload, and rolling-restart acceptance runs remain deliberately unexecuted.
  Record a genuinely trained checkpoint version/checksum and prove fallback
  count zero before treating future results as production acceptance evidence.

## 2026-07-30 — Visual temporal attention recall path

- Added an independent `VisualRetrievalPooler` that temporal-encodes up to 16
  real-timestamped frame CLIP vectors, scores frames, weights the original
  frozen 512-d CLIP vectors, and mixes them with mean pooling through a
  near-zero initialized learned gate.
- Added PIT-safe weighted multi-positive InfoNCE training with 1/2/3
  click/cart/purchase weights, same-content false-negative masking, unclicked
  impression negatives, time-group validation, optional ranker visual-encoder
  warm start, and a strict `visual_retrieval_attention_v1` checkpoint lineage.
- Added offline real-product-image CLIP encoding, checksum quarantine, a 95%
  coverage gate, HNSW inner-product FAISS bundles, atomic manifest activation,
  artifact persistence/sync, and removal of random fallback from this new
  production index path.
- Added `temporal_multimodal_v3`, preserved mean and attention retrieval
  vectors, offline content-worker inference, no-video-decode v2-to-v3 backfill,
  deterministic canary serving, strict CLIP/index compatibility, pinned CLIP
  revision, deployment settings, and fixed-index Recall/MRR/coverage evaluation.
- Key files: `video_commerce/ml/visual_retrieval.py`,
  `video_commerce/ml/visual_product_index.py`,
  `video_commerce/ml/content_processor.py`,
  `video_commerce/services/model_trainer/main.py`,
  `scripts/build_visual_product_index.py`, and
  `scripts/backfill_visual_retrieval.py`.
- Verification: focused Docker suite 49 passed; expanded visual/content/PIT/
  trainer/artifact/config suite 130 passed; ranking optimizations 77 passed;
  full Docker backend regression 546 passed, 8 skipped. Compose config, Helm
  lint, kubeconform (33 valid), Black, focused Flake8, and `git diff --check`
  passed. Real catalog coverage, Recall uplift confidence bounds, and
  50/100/500-candidate latency promotion gates still require production data
  and therefore are not claimed.
- Final reviewer findings were fixed by observation-time filtering product-index
  hard negatives, masking same-product and all-content-positive false negatives,
  and giving visual shadow training an independent success marker/retry path
  outside the main ranking manifest duplicate gate.

## 2026-07-12 — PIT DIN ranking implementation

- Added `din_sequence_v1` click/cart/purchase histories with strict 30-day PIT
  filtering, chronological left-padding to 60, repeated-event preservation,
  action-specific Redis ZSETs, and sequence-aware cache freshness.
- Added candidate-aware DIN attention over a checksummed, ranking-owned frozen
  128-d two-tower sidecar; ranking keeps the existing CTR/CVR/CTCVR/GMV and LTR
  objectives and now records AUC, NDCG, attention entropy, sequence coverage,
  embedding coverage, and unknown-item rate.
- Added `ranking_ltr_v2_din`, `ranking_feature_assembler_v2_din`,
  `ranking_v3_din`, PIT `behavior_sequences_json`, payload v3 capability
  negotiation, request-level history reuse, a 30% non-empty sequence gate, and
  jointly checksummed checkpoint/embedding-sidecar activation metadata with
  strict load-time lineage validation.
- Key files: `video_commerce/ml/din.py`, `video_commerce/ml/ranking.py`,
  `video_commerce/ml/pit_training_dataset.py`,
  `video_commerce/ranking_runtime/ranking_batcher.py`,
  `video_commerce/services/model_trainer/main.py`, and the Flink interaction/PIT
  jobs.
- Verification: Docker backend suite 472 passed, 4 skipped; Flink Maven 27
  passed; targeted Black, `docker compose config -q`, and Helm lint passed.
  After merging the temporal multimodal ranking work from `main`, the combined
  DIN/batcher/temporal/ranking optimization Docker suite passed 113 tests.
  Independent review findings were fixed for strict checkpoint/sidecar lineage,
  atomic checkpoint writes, v3 fail-closed negotiation, stable history reuse,
  unknown-item zeroing, Iceberg schema evolution, corrupt Redis reads,
  full-sequence history identity, score-bounded ZSET reads, and explicit
  combined-score fallback when official DIN history is unavailable; the
  degraded path also bypasses recommendation-cache reads and writes.
  CPU DIN-branch p95 with 10 valid
  events/action was 26.5 ms at 100 candidates and 48.3 ms at 250 candidates.
- Follow-up/blocker: the approved +10 ms p95 activation budget is not met on the
  current CPU image, so DIN remains disabled by default and must not be activated
  until the production-target benchmark passes or the attention path is further
  optimized. The dedicated independent reviewer completed final re-review with
  `Ready to merge: Yes`.

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
  presence-aware modality gating, batch-local 10% OCR/ASR dropout, V3
  warm-start, configurable base-freeze warm-up, differential learning rates,
  time-based impression validation, best-state restoration, and the
  `ranking_v4_01_temporal_trimodal` checkpoint schema. DIN and trimodal tensors
  now share one structured training-batch contract and loss-contract errors
  fail fast.
- PIT manifests now pin and verify candidate sidecar URI, SHA-256, schema,
  source model version, and source-model training cutoff. Candidate embeddings
  are versioned by `available_at` and resolved per observation, preventing
  later item versions from leaking into earlier samples. PIT and default
  trimodal-shadow training require pinned sidecars and do not read current
  FAISS/CF state. Training history records modality/candidate coverage, gate
  means, temporal-attention entropy, encoder gradient norms, validation loss,
  and completed epochs.
- Added bounded payload V4 support across recommendation, ranking service,
  coordinator, runner, and local fallback; checksum-locked candidate NPZ
  sidecars with atomic activation; V1-V3 compatibility; shadow-by-default
  rollout and V3 checkpoint rollback lineage. V4 activation now rejects
  incomplete state dictionaries, shadow config is fully revalidated, and the
  post-holdout training partition must still satisfy the sample threshold.
  Artifact publication is gated by an explicit per-attempt training result, so
  a skipped main or shadow run cannot republish previously loaded weights with
  new PIT lineage.
- Key files: `asr_service/api.py`, `video_commerce/ml/content_processor.py`,
  `video_commerce/ml/content_artifacts.py`,
  `video_commerce/ml/temporal_multimodal.py`, `video_commerce/ml/ranking.py`,
  `video_commerce/ml/ranking_training.py`, and
  `video_commerce/ml/candidate_embedding_sidecar.py`.
- Verification: after the training-correctness review fixes, the expanded
  temporal/training/PIT/ranking Docker suite passed 231 tests with 3 skips; the
  complete backend suite passed 524 tests with 8 skips. Python compile, scoped
  Black, diff checks, `docker compose config -q`, Helm lint, and strict
  kubeconform passed for both default (33 resources) and PIT production
  (36 resources) renders. GPU 300-second ASR profiling, v3/v4 candidate
  benchmarks, and offline NDCG/AUC/GMV promotion evaluation require production
  hardware/data and remain rollout gates. No performance improvement is
  claimed.

## 2026-10-09 — Ranking quality gate and release lifecycle

- Added additive Postgres release, evaluation, pointer, and transition tables;
  checkpoint persistence now creates an immutable `registered` ranking bundle
  and repairs release registration after a checkpoint/outbox-style crash race.
- Reserved a fixed seven-day mature PIT holdout before training and value
  normalization. The trainer compares the active champion and challenger over
  identical rows with canonical serving scores, deterministic user-cluster
  bootstrap confidence intervals, and cold-user, cold-item, long-tail,
  high-price, and modality-missing slices. Passing releases advance
  automatically through `validated` to `staging`; failed or insufficient
  releases remain non-serving.
- Added audited CAS promotion, rollback, and one-time verified bootstrap CLI;
  production serving selects only the active pointer in enforced mode. Legacy
  runners carry release/generation lineage into durable impressions, and Triton
  startup requires its exact model version to match the active release.
- Added release backlog, staging age, evaluation decision, active generation,
  and runner mismatch metrics and alerts, plus Compose observe-mode defaults,
  enforced Helm production settings, and the model release rollout/rollback
  runbook.
- Key files: `migrations/postgres/010_model_release_quality_gate.sql`,
  `video_commerce/ml/model_release.py`,
  `video_commerce/ml/model_release_quality.py`,
  `video_commerce/ml/model_artifacts.py`,
  `video_commerce/services/model_trainer/main.py`, and
  `docs/operations/model-release-runbook.md`.
- Verification: isolated Docker backend suite passed 687 tests with 5 skips;
  the Postgres lifecycle smoke proved bootstrap, pass-to-staging, CAS promotion,
  stale-generation rejection, rollback eligibility, and failed-challenger
  isolation. Production backend image build, Compose validation, Helm lint,
  strict kubeconform (36 resources), Prometheus rule tests, scoped Black,
  Python compile, and diff checks passed. No online canary or performance claim
  is included; assignment, propensity, SRM, and uplift gates remain a later
  phase.
