# Model artifact rolling activation

Ranking and Two-Tower production serving resolve the active release pointer, install every bundle into an immutable generation directory, validate all checksums and contracts, and load a fresh model or retrieval engine before publishing it to new requests. The generation identity is `(release_id, active_generation, bundle_digest)`; a reused model version does not suppress a required reload.

Set `MODEL_ARTIFACT_GENERATION_ROOT` to pod-local scratch storage. Services report `loading`, `ready`, and `failed` observations in `model_release_instance_observations`. A `ready` observation is written only after inference loading and smoke validation. A failed replacement leaves the previous loaded instance serving.

After promotion, inspect convergence:

```bash
python scripts/model_release.py --model-name ranking_model rollout-status --generation 12 --json
python scripts/model_release.py --model-name ranking_model wait-rollout \
  --generation 12 \
  --expected-service-count ranking-service=2 \
  --expected-service-count ranking-runner=4 \
  --expected-service-count recommendation-service=2 \
  --timeout 300
```

Recommendation re-reads release pointers every
`RANKING_CHECKPOINT_SYNC_INTERVAL_SECONDS`, in every topology. Each cycle it
also hot-swaps retrieval artifacts (Two-Tower, SASRec, Swing ItemCF, content
clusters, vector index). It then rebuilds the serving version context, which
feeds the recommendation cache key and the impression lineage
(`model_release_id`, `active_generation`, `quality_gate_policy_version`).

When ranking runs remotely (coordinator or ranking-service, no local ranker),
recommendation has no loaded model to read. Its lineage comes from the active
ranking pointer in Postgres, so a promotion changes the cache key and impression
lineage within one poll interval. During a rolling activation that pointer can
lead what some runners have actually loaded, for up to one runner poll interval
plus model load time. Impressions served in that window carry the new release
id while some slates were scored by the previous generation. Use
`wait-rollout` above before attributing a metric change to the new release.
When the pointer lookup fails, the last known release is kept.

Local cleanup retains the current generation, all pointer-referenced generations, in-flight batch generations, the configured newest generations, and every generation younger than the minimum age. Do not manually delete generation directories while a service is running. Rollback changes the durable pointer through `model-release rollback`; each instance then installs and validates that generation through the same path.

When `RANKING_INFERENCE_PROCESS_WORKERS` is non-zero, use a rolling restart for activation. Forked process workers cannot safely replace their initialized model in place, so the service leaves the previous generation active and reports rollout failure rather than mixing processes.
