# Retrieval PIT And Two-Tower Release Runbook

This runbook governs collaborative Two-Tower retrieval. Production training
uses an immutable retrieval PIT dataset and production serving uses only the
release selected by the active `two_tower_retrieval` pointer. Recommendation
processes must use `RECOMMENDATION_ONLINE_RETRAINING_ENABLED=false`.

## Build the pinned inputs

Create a complete catalog snapshot JSON containing `products` and `embeddings`.
Every product belongs to the generation even when an optional modality is
missing; use a missing embedding value rather than dropping the product.

```bash
python scripts/build_retrieval_catalog_generation.py \
  --input /secure/catalog-snapshot.json \
  --generation-id catalog-20261009 \
  --effective-at 1791511200 \
  --available-at 1791514800 \
  --embedding-model-revision clip-pinned-revision \
  --output-reference /secure/catalog-20261009-reference.json
```

Submit the batch materializer with a unique run ID and attempt. The cutoff must
be after the 168-hour attribution window and allowed lateness for every viewed
query that should be finalized.

```bash
flink run \
  -c com.videocommerce.flink.RetrievalPointInTimeJoinJob \
  /opt/flink/usrlib/interaction-features.jar \
  --materialization-run-id retrieval-pit-20261009 \
  --catalog-generation-id catalog-20261009 \
  --materialization-cutoff 1791514800 \
  --export-uri s3://video-commerce-features/training/retrieval-pit \
  --export-attempt 1
```

Publish `latest.json` only after all Parquet shards and both catalog artifacts
exist. The publisher verifies schemas, row count, catalog generation, and every
SHA-256 before moving the pointer.

```bash
python scripts/publish_retrieval_pit_manifest.py \
  --shard s3://video-commerce-features/training/retrieval-pit/retrieval-pit-20261009/attempt-1/part-00000.parquet \
  --run-id retrieval-pit-20261009 \
  --dataset-version catalog-20261009 \
  --attribution-cutoff 1791514800 \
  --catalog-reference /secure/catalog-20261009-reference.json
```

## Train and evaluate

The trainer reserves the latest seven mature days, fits mappings, frequency,
normalization, and negative mining from the pre-holdout partition, then records
popularity, active champion, and challenger results. Evaluation uses the full
eligible ANN catalog and a deterministic exact-dot-product audit subset.

```bash
RETRIEVAL_TRAINING_SOURCE=pit \
FEATURE_LAKE_RETRIEVAL_PIT_DATASET_URI=s3://video-commerce-features/training/retrieval-pit/latest.json \
python scripts/train_retrieval_pit.py --model-version two-tower-20261009 --json
```

The default negative policy uses viewed negatives, weak ranker-rejected
examples, ANN hard negatives, and uniform negatives. Controlled experiments
must reuse the same PIT manifest, catalog generation, seed, and training budget:

- control: `RECOMMENDATION_TT_LOGQ_CORRECTION_ENABLED=false`
- logQ: `RECOMMENDATION_TT_LOGQ_CORRECTION_ENABLED=true`
- rejected disabled: `RETRIEVAL_RANKER_REJECTED_MODE=disabled`
- rejected weak: `RETRIEVAL_RANKER_REJECTED_MODE=weak`
- teacher soft target: `RETRIEVAL_RANKER_REJECTED_MODE=teacher_soft`

Use a distinct model version per treatment. The release manifest records the
negative policy, mining catalog generation, and mining cutoff. Do not compare
runs whose PIT, catalog, seed, or training budget differs.

## Promote, observe, and roll back

A passing evaluation advances a release to `staging`. It does not change
serving. Inspect evidence, then promote with the current active generation:

```bash
python scripts/model_release.py --model-name two_tower_retrieval status --json
python scripts/model_release.py --model-name two_tower_retrieval promote \
  --version two-tower-20261009 \
  --expected-generation 3 \
  --actor release-manager \
  --reason "retrieval_quality_gate_v1 passed"
```

For the first governed release only, use `bootstrap-active` after validating
the complete bundle. It records `bootstrap_uncompared=true`.

```bash
python scripts/model_release.py --model-name two_tower_retrieval bootstrap-active \
  --version two-tower-bootstrap \
  --actor release-manager \
  --reason "initial verified PIT retrieval bundle"
```

Serving downloads the desired generation, verifies all checksums, schema,
dimension, policies, mapping metadata, and checkpoint load, then swaps the
in-process engine. A failed reload keeps the previous in-memory generation.
Production readiness fails on restart when enforced mode cannot load a valid
active release.

Rollback uses the same generation fence and only accepts a compatible release
that previously passed its gate:

```bash
python scripts/model_release.py --model-name two_tower_retrieval rollback \
  --version two-tower-previous \
  --expected-generation 4 \
  --actor oncall \
  --reason "retrieval regression"
```

Monitor `retrieval_release_registered_backlog`,
`retrieval_release_staging_age_seconds`, and
`retrieval_release_active_generation`. Durable impression payloads include the
Two-Tower release ID and active generation used for candidate retrieval.

## Evidence boundary

Synthetic fixtures verify engineering behavior only. A production-quality
baseline requires a real complete catalog generation and mature labels. If
overall or required slice evidence is below the fixed gate, the decision is
`insufficient_evidence`; do not bootstrap or claim a retrieval quality gain.
