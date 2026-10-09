# Ranking model release runbook

Ranking serving is governed by a durable release registry. Persisting a new
checkpoint creates a `registered` release; it does not change production
serving. The model trainer reserves the latest seven mature PIT days as a fixed
holdout, compares the challenger with the current active release, records the
versioned `ranking_quality_gate_v1` evidence, and moves passing bundles to
`staging`. Activation remains a manual, audited operation.

## Deployment order

1. Apply `migrations/postgres/010_model_release_quality_gate.sql` after the
   preceding additive migrations.
2. Deploy readers in `MODEL_RELEASE_GATE_MODE=observe`. They prefer an active
   pointer and retain the legacy latest-checkpoint fallback while the first
   release is registered.
3. Inspect the registered bundle and materialize its exact checkpoint, ONNX,
   candidate sidecar, and DIN sidecar. Checksums, feature schema, score policy,
   and value-normalization metadata must all pass.
4. Bootstrap once when no active pointer exists:

   ```bash
   python scripts/model_release.py bootstrap-active \
     --version VERSION --actor OPERATOR --reason "initial verified release"
   ```

5. Switch production to `MODEL_RELEASE_GATE_MODE=enforced`. Readiness now fails
   if the active pointer or compatible bundle is absent.

## Evaluate and promote

The trainer evaluates each new PIT challenger automatically. The CLI reports
the durable outcome:

```bash
python scripts/model_release.py evaluate --version VERSION
python scripts/model_release.py status --json
```

Only a `staging` release with a current passing evaluation can be promoted.
Use the current active generation from `status` as the CAS fence:

```bash
python scripts/model_release.py promote \
  --version VERSION \
  --expected-generation GENERATION \
  --actor OPERATOR \
  --reason "offline quality gate passed"
```

The active pointer update, retirement of the previous active release, and
transition audit are one Postgres transaction. Legacy runners download and
verify the selected generation before reload. Triton deployments must set
`RANKING_REQUIRED_MODEL_VERSION` to the same active model version, then perform
the normal Helm rolling restart.

## Rollback

Rollback can target only a compatible release that previously passed the gate
and is now retired. Read the current generation immediately before issuing the
rollback:

```bash
python scripts/model_release.py rollback \
  --version PREVIOUS_VERSION \
  --expected-generation GENERATION \
  --actor OPERATOR \
  --reason "serving regression"
```

If the generation changed, inspect status and the transition audit before
retrying. Do not edit pointer rows manually. A rollback changes only the active
ranking bundle; additive registry and evaluation tables stay in place.

## Monitoring

Watch `model_release_registered_backlog`,
`model_release_staging_age_seconds`, `model_release_active_generation`,
`model_release_runner_generation_mismatch`, and
`model_release_evaluations_total`. A generation mismatch lasting ten minutes is
critical. Registered or staging releases older than a day require a decision or
an investigation of PIT evidence and artifact validation.
