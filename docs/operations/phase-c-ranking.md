# Phase C ranking experiments and activation

Phase C adds the `ranking_ltr_v3_categorical_sources` model input contract,
source-aware candidate fusion, `gross_revenue_v1` value labels, and optional
`temperature_v1` calibration. Production retains legacy fusion and the existing
active model until an operator promotes a fully evidenced combined release.
DIN, trimodal, SASRec and MMoE are separate experiments.

## Data and feature contracts

The raw Flink feature-history/PIT contract remains `ranking_ltr_v1`. It now
preserves category, brand, tags, request context, per-source retrieval signals,
and attributed purchase value, GMV and currency as additive fields. The learned
model input contract is v3; do not change the raw materializer contract to v3.
Older shards without currency retain purchase labels but have masked value
labels under `gross_revenue_v1`.

Only matching-currency `purchase_value` or `gmv` contributes to value loss.
Conflicting values are masked and counted; margin, profit and product price are
not substitutes. Production must explicitly set `RANKING_VALUE_CURRENCY`.

The latest seven mature PIT days are the holdout. Before that boundary, the
latest 10% of impression groups form calibration; early-stopping validation is
selected from the remaining pre-holdout groups. Training alone determines
vocabularies and clipped p1/p99 population normalization. Entire impression
groups stay together across shards and splits. SQLite accumulators bound
vocabulary and normalization memory; calibration accumulates a fixed likelihood
grid rather than retaining predictions.

V3 freezes PAD=0, MISSING=1, OOV=2 and deterministic token vocabularies. Category,
brand and pooled tags/preferences use 16-dimensional embeddings; device,
surface and UTC time-of-day use eight dimensions. Packed integer IDs share the
existing FP32 ranking batch protocol, then become learned embeddings inside
Torch/ONNX. Bundle and checkpoint vocab/normalization must agree.

`retrieval_sources_v1` maps aliases to nine known sources plus unknown. Legacy
fusion retains its existing ranking/diversity behavior and records all source
signals. RRF sums `1/(60 + source_rank)`, sorts ties by product ID, retains at
most 500 candidates, and does not apply a category cap before ranking. MMR
still runs after ranking. Raw scores remain source-specific audit values.

## Produce independent bundles

Use one immutable ranking PIT pointer, the same verified catalog, seed 42 and
independent model versions. The following creates and registers a feature-only
release; registration does not change the active pointer:

```bash
PYTHONPATH=. python scripts/train_phase_c.py \
  --pit-pointer /data/ranking-pit/latest.json \
  --experiment features --model-version phase-c-features-1 \
  --output-dir /data/phase-c --register
```

Repeat with `--experiment combined` for the combined v3/temperature bundle.
`fusion` clones the pinned champion without an optimizer; `calibration` clones
it and fits temperatures only on reserved calibration groups. Both require
`--champion-bundle` and `--champion-bundle-sha256`. Feature/combined training uses
v3 with DIN/trimodal disabled; feature-only disables calibration.

The CLI saves checkpoint, parity-checked ONNX and a pinned bundle JSON. Its
`training_pit_materialization_run_id` records data provenance without reusing the
production trainer's single-winner `feature_lake_materialization_run_id` claim;
independent model versions can therefore compare the same PIT run. Failed or
insufficient runs never activate a release. Inspect and remove unpublished
partial output directories before retrying with a fresh version.

## Evaluate and preserve evidence

First obtain real retrieval baseline evidence from the existing retrieval PIT
trainer/evaluator: popularity, active Two-Tower and challenger on the full
eligible catalog, mature labels, Recall@50/100/200, MRR and coverage. Synthetic
fixtures do not satisfy this requirement.

Candidate Recall@500 evidence must be supplied from a real candidate-pool
comparison; the ranking evaluator does not invent unseen candidates from
served rows. The evidence JSON includes `pit_manifest_sha256`,
`catalog_manifest_sha256`, `champion_bundle_digest`, `challenger_bundle_digest`,
`candidate_limit: 500`, `seed: 42`, `champion_recall_at_500` and
`challenger_recall_at_500`. Missing evidence produces `insufficient_evidence`.
Here a bundle digest is SHA-256 of sorted compact JSON of the immutable registry
bundle manifest. It is distinct from the generation installer's local digest.

Replace digest placeholders with the SHA-256 of each JSON file:

```bash
PYTHONPATH=. python scripts/evaluate_phase_c.py \
  --pit-pointer /data/ranking-pit/latest.json \
  --champion-bundle /data/champion.json --champion-bundle-sha256 CHAMPION_JSON_SHA \
  --challenger-bundle /data/phase-c/phase-c-features-1/bundle.json \
  --challenger-bundle-sha256 CHALLENGER_JSON_SHA \
  --experiment features --candidate-recall-evidence /data/recall-features.json \
  --output /data/evaluations/features.json --register
```

Evaluate features, fusion and calibration separately. Each has its own v2
improvement threshold. Combined evaluation repeats the full gate and adds
`--prerequisite-evaluation-id features=UUID`, `fusion=UUID`, `calibration=UUID`.
Promotion checks that all three passed using the same champion, PIT and catalog.
All comparisons use the same holdout rows, canonical score and deterministic
product-ID tie breaking. ONNX output scores/top-k are checked on holdout batches.
Passed evaluation may stage a bundle; it never promotes it.

Evaluation records are immutable per release/PIT/policy. Do not reinterpret an
existing insufficient/failed evaluation with new thresholds. Use a new model
version when evidence or an experiment changes.

`ranking_quality_gate_v4` slice policy:

- Only the pre-registered slices block the gate: `cold_user`, `cold_item`, and
  `price_low`/`price_mid`/`price_high`. Their NDCG CIs are Bonferroni-adjusted
  across the blocking slices that are tested.
- Category and retrieval-source slices are capped to the top 10 by impressions,
  plus an `other` bucket. Missing-modality slices are also report-only. All of
  these are recorded with `"blocking": false` and never change the decision.

`ranking_quality_gate_v5` NDCG decision (v5 evidence stays valid for
promotion):

- NDCG@10 is a non-inferiority test on the user-cluster bootstrap CI against
  the margin (`overall_ndcg_ci_lower` = -0.005; blocking slices -0.03 with the
  Bonferroni-adjusted CI):
  - The CI lower bound is at or above the margin: the NDCG check passes.
  - The whole CI is below the margin: `overall_ndcg_regression` (or
    `slice_<name>_ndcg_regression`) and the gate fails.
  - The CI straddles the margin: `overall_ndcg_underpowered` (or
    `slice_<name>_ndcg_underpowered`) and the gate is `insufficient_evidence`.
    The holdout is too small to decide. `ndcg_informative_impressions_required_estimate`
    gives a rough holdout size (`n × (half_width / (delta − margin))²`) at
    which the observed delta would clear the margin. It is null when the point
    delta is already below the margin.
- The point-delta check applies only to a positive `*_ndcg_min_delta`
  (superiority). The default minimum of 0 is a non-inferiority requirement,
  which the CI test covers. Under v4, an equal-quality challenger failed that
  check about half the time.
- `calibration_ndcg_regression` fires only when the NDCG CI is wholly below 0.
- Definite failures take precedence over underpowered results.
- Under v5, PR-AUC, log loss, Brier, ECE and value WAPE were still point
  thresholds, so small holdouts failed equal-quality challengers on them.

`ranking_quality_gate_v6` CI decision for every metric (current policy;
v2–v5 evidence stays valid for promotion):

- Every gated metric uses the v5 NDCG three-way rule on its own user-cluster
  bootstrap CI. All metrics share the same resampled users in each sample.

  | Metric | Threshold (overall / slice) | Fails when |
  | --- | --- | --- |
  | `ndcg_at_10_delta` | -0.005 / -0.03 | CI upper < threshold |
  | `{ctr,ctcvr}_pr_auc_delta` | -0.005 / -0.005 | CI upper < threshold |
  | `{ctr,ctcvr}_log_loss_relative_regression` | 0.02 / 0.05 | CI lower > threshold |
  | `{ctr,ctcvr}_brier_relative_regression` | 0.02 / 0.05 | CI lower > threshold |
  | `{ctr,ctcvr}_ece_absolute_regression` | 0.005 / 0.01 | CI lower > threshold |
  | `value_wape_relative_regression` | 0.05 / 0.05 | CI lower > threshold |

  A CI wholly beyond its threshold fails with the v5 reason name, for example
  `overall_ctr_log_loss_relative_regression` or `slice_price_mid_ndcg_regression`.
  A CI straddling it is `insufficient_evidence` with
  `<prefix>_<metric>_underpowered`, for example `overall_ctr_ece_underpowered`
  or `slice_price_mid_ctcvr_pr_auc_underpowered`. Otherwise the metric passes.
- Each gated metric reports `<metric>_ci_lower`, `_ci_upper` and
  `_ci_half_width` in `overall`, in every evaluated slice and in `bootstrap`.
- Blocking slices use one Bonferroni-adjusted alpha for all of their metrics.
- The superiority requirements are unchanged point checks: features and
  fusion minimum improvement, calibration ECE improvement and no Brier
  regression, and a positive `*_ndcg_min_delta`. Candidate recall@500 is
  external evidence and is still a point threshold.
- The ECE CI keeps each row in its equal-frequency bin from the evaluated
  rows. Resamples reweight users within those bins and do not re-bin.
- A slate is one user's rows of one impression. This matches v5 whenever every
  impression belongs to one user.
- Small holdouts now usually return `insufficient_evidence`. 240-impression
  A/A simulations ran 160 equal-quality runs at score noise 0.01 and 0.05.
  None failed, and 159 were underpowered, mostly on ECE, ctcvr PR-AUC and
  NDCG. An underpowered result is not a regression: collect a larger holdout
  before you decide.
- Bootstrap cost: per-user sums are computed once. Each sample is a
  multinomial user weight vector, and samples are processed in chunks as matrix
  products. PR-AUC is a weighted average precision over one sort. On a laptop,
  the overall bootstrap for 1M rows × 2000 samples takes about 26 s with a
  300 MB traced peak, which does not grow with the sample count. With 632k
  single-row users it takes 70 s. The full gate on 1M rows with 29 slices
  takes about 2.2 minutes; v5 needed hours. Run the benchmark with
  `RANKING_GATE_BENCHMARK=1 python -m pytest -s tests/test_model_release_quality.py -k million`.
  A seed draws the same resamples as v5, so v6 NDCG CIs equal v5's.

The general trainer evaluates every periodic retrain with experiment kind
`retrain`:

- The champion is the active release, loaded from its own installed checkpoint
  and refreshed every round. Challengers go to `<ranking_model>.challenger.pt`.
- Candidate recall is not required, because the candidate generator does not
  change.
- A pass may stage the release. Activation still requires the full Phase C
  evidence from these CLI runs (prerequisites and ONNX parity).
- Activation rejects any evaluation whose `champion_release_id` is not the
  current active release (`champion_not_active`).

## Performance proof and promotion

Run the full gateway/recommendation path on fixed Linux CPU/memory allocations,
comparing against the active champion in hot and unique modes. Archive raw
results. Performance proof JSON must contain the target `release_id`, registry
`bundle_digest`, raw `results_sha256`, `platform: linux`, and
`resources: {cpus, memory_bytes}`. Both mode entries must include fractional
`error_rate`, `five_xx_rate`, millisecond `successful_p95_ms`,
`successful_p99_ms`, and relative `p95_relative_regression`,
`successful_qps_relative_regression` (positive means regression).

The gate requires error <0.5%, 5xx <0.1%, p95 <1000ms, p99 <2000ms and both
regressions <=10%. Do not write placeholder evidence for missing measurements.
The CLI verifies the raw result file SHA before passing evidence to the
transactional promotion gate. The proof is archived in transition audit data.

```bash
PYTHONPATH=. python scripts/model_release.py promote \
  --version phase-c-combined-1 --expected-generation GENERATION \
  --actor OPERATOR --reason "Phase C offline and Linux gates passed" \
  --performance-evidence /data/performance-proof.json \
  --performance-results /data/raw-performance.json
```

V3 cannot use bootstrap-active. Rollback to a previously active compatible
release uses CAS; its archived performance proof can be reused. Production RRF
startup requires an active promoted v2 bundle with `candidate_rrf_v1`, k=60 and
`retrieval_sources_v1`. Deploy model and fusion configuration together. Observe
rollout generations and preserve the prior verified bundle on load failure.

## Current acceptance boundary

Unit/contract tests and synthetic ONNX parity establish engineering behavior.
Real ranking/retrieval baselines, the four challengers, Linux performance,
Postgres release integration and the isolated release smoke are separate
acceptance evidence. Without them, retain legacy fusion/active selection and
report Phase C as pending acceptance rather than model improvement.
