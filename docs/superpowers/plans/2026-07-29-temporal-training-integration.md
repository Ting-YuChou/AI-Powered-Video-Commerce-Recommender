# Temporal Encoder Training Integration Plan

Date: 2026-07-29
Branch: `codex/temporal-training-integration`
Base: latest `origin/main` (`fac1a0f`)

## Objective

Make the temporal visual/OCR/ASR encoders a fully governed part of ranking
training rather than only a reachable model graph. A single PIT-pinned training
batch must support:

```text
base ranking features
+ optional DIN behavior inputs
+ visual/OCR/ASR temporal sequences
+ candidate-conditioned attention
-> CTR/CVR/CTCVR/value/listwise/pairwise objectives
```

The completed path must train, validate, checkpoint, reload, shadow-serve, and
fail closed with the same feature and artifact contracts.

## Scope and boundaries

- Keep FAISS content recall on the compatibility mean CLIP embedding.
- Keep CLIP, multilingual E5, Qwen ASR/aligner, and two-tower producers frozen.
- Keep public gateway APIs unchanged.
- Preserve payload v1-v4 compatibility.
- Keep trimodal serving disabled by default until offline and latency gates
  pass.
- Preserve base-only behavior when content or candidate modalities are absent.

## Phase 1: Training correctness

### 1.1 Unified training tensor contract

Files:

- `video_commerce/ml/ranking_training.py`
- `video_commerce/ml/ranking.py`
- `tests/test_ranking_training_contract.py`

Changes:

1. Introduce structured `RankingTrainingTensors` and `RankingTrainingBatch`
   contracts with separate `labels`, `trimodal_inputs`, and `din_inputs`.
2. Remove the hidden `_din_*` tensors from the label dictionary.
3. Build DIN and trimodal inputs independently so all four configurations work:
   base only, DIN only, trimodal only, and DIN plus trimodal.
4. Slice every input group through one batch iterator.

Acceptance:

- Every configuration completes one optimizer step.
- DIN plus trimodal training has no missing-key path.
- Labels contain supervised targets only.

### 1.2 Single residual gate and safe missing-candidate behavior

Files:

- `video_commerce/ml/temporal_multimodal.py`
- `video_commerce/ml/ranking.py`
- `tests/test_temporal_multimodal.py`

Changes:

1. Remove the inner residual scale from `TrimodalCandidateAttention`.
2. Keep one outer near-zero gate in `TemporalTrimodalRankingModel`.
3. Mask the trimodal residual to zero when all candidate embeddings are absent.
4. Preserve zero residual when all content modalities are absent.
5. Bump the model architecture/checkpoint contract to
   `ranking_v4_01_temporal_trimodal`.

Acceptance:

- Initial output remains close to the base ranker.
- Each present temporal encoder receives finite non-zero gradients.
- All-candidate-missing and all-content-missing rows exactly follow base-only
  behavior.
- Old V4 shadow checkpoints are not activation-compatible with V4.01.

### 1.3 Fail-fast loss and governed training schedule

Files:

- `video_commerce/common/config.py`
- `video_commerce/ml/ranking.py`
- `tests/test_ranking_training_contract.py`

Changes:

1. Remove the catch-all zero-loss fallback.
2. Add configurable warm-up epochs, minimum epochs, validation fraction,
   early-stopping patience, and minimum delta.
3. Do not allow early stopping during frozen-base warm-up.
4. Apply OCR/ASR modality dropout per training batch/epoch, not once when the
   entire dataset tensor is built.
5. Make dropout deterministic under the configured training seed.

Acceptance:

- Invalid loss inputs stop training and prevent artifact publication.
- The base ranker is unfrozen before early stopping is eligible.
- Validation receives no modality dropout.
- Identical seed/manifest/checkpoint inputs reproduce the same dropout and
  predictions.

## Phase 2: Evaluation parity

Files:

- `video_commerce/ml/ranking.py`
- `video_commerce/services/model_trainer/main.py`
- `tests/test_ranking_training_contract.py`

Changes:

1. Split PIT examples by time while keeping an impression in one split.
2. Pass the complete base, trimodal, and DIN inputs into evaluation.
3. Select and restore the best validation checkpoint.
4. Record validation loss, NDCG@10, CTR AUC, value metrics, modality coverage,
   gate distributions, attention entropy, and per-encoder gradient norms.
5. Add base/visual/visual+OCR/full-trimodal ablation evaluation.

Acceptance:

- Changing valid temporal inputs can change validation predictions and metrics.
- Masking all temporal inputs reproduces base predictions.
- Reported trimodal AUC/NDCG never use the base-only shortcut.
- Train and validation impressions do not overlap.

## Phase 3: PIT and candidate lineage

Files:

- `video_commerce/ml/pit_manifest.py`
- `video_commerce/ml/pit_training_dataset.py`
- `video_commerce/ml/candidate_embedding_sidecar.py`
- `video_commerce/ml/model_artifacts.py`
- `video_commerce/services/model_trainer/main.py`
- related PIT/artifact tests

Changes:

1. Add candidate image/text/two-tower source references and checksums to the PIT
   training manifest.
2. Pin E5 model, revision, prefix, and dimension.
3. Stop trimodal PIT training from reading unpinned current vector-search or CF
   sidecar state.
4. Validate content artifacts, candidate sources, timestamps, dimensions, and
   model revisions before tensor construction.
5. Keep no-reference rows available for base-only training, but fail closed for
   invalid references.

Acceptance:

- Candidate feature construction is reproducible from one manifest.
- Checksum, schema, revision, or dimension mismatch stops training.
- Checkpoint, candidate sidecar, and PIT lineage form one atomic activation
  unit.

## Phase 4: Data readiness and backfill

Files:

- `scripts/backfill_temporal_multimodal.py`
- content artifact and observability modules
- operations documentation

Changes:

1. Dry-run and then enqueue `temporal_multimodal_v2` backfill in an ASR-capable
   environment.
2. Report content-reference, visual, OCR, ASR, and candidate embedding coverage.
3. Keep degraded ASR transcript metadata but mask the temporal ASR branch.
4. Add configurable data-readiness gates before trimodal training.

Initial gates:

- Valid visual sequence for at least 95% of referenced content examples.
- Candidate text embedding for at least 99% of candidates.
- All present timestamps/embeddings/checksums valid.
- OCR/ASR presence reported as slices rather than mandatory coverage.

## Phase 5: Promotion and rollout

1. Train shadow artifacts with `RANKING_TRIMODAL_SHADOW=true` and serving
   disabled.
2. Compare base, visual, visual+OCR, full trimodal, and DIN+trimodal on a
   time-based holdout.
3. Require:
   - NDCG@10 does not decline.
   - CTR AUC declines by no more than 0.002.
   - Value/GMV proxy does not decline.
   - Ranking requests have zero errors/rejections.
   - V4 p95 is no more than 20% above V3 at 50/100/500 candidates.
4. Canary at 1%, 10%, 50%, and 100% only after gates pass.
5. Keep the prior checkpoint reference for immediate rollback.

## Verification

Focused host tests:

```bash
python -m pytest -q \
  tests/test_temporal_multimodal.py \
  tests/test_ranking_training_contract.py \
  tests/test_candidate_embedding_sidecar.py \
  tests/test_pit_training_dataset.py \
  tests/test_ranking_optimizations.py
```

Repository and deployment validation:

```bash
python -m pytest -q
docker compose --profile test run --rm backend-tests
docker compose config -q
docker run --rm -v "$PWD:/work" -w /work alpine/helm:3.14.0 \
  lint charts/video-commerce
```

After any OrbStack-backed validation:

```bash
docker compose down --remove-orphans
docker compose ps
```

Do not delete volumes. Do not claim latency or model-quality improvements until
the matching production-data/hardware evaluations have run.

## Implementation order

1. Unified batch contract and combined DIN/trimodal regression test.
2. Single residual gate and missing-candidate fallback.
3. Fail-fast loss, warm-up, dynamic dropout, and validation parity.
4. PIT candidate lineage and artifact contract.
5. Coverage, ablation, promotion, and rollout automation.
6. Documentation, `PROGRESS.md`, scoped verification, commit, push, and PR.
