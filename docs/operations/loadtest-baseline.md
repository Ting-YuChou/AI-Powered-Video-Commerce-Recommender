# Loadtest Baseline

## Script
- `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode hot --timeout 10`
- `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode unique --timeout 10`

Use `--run-id` to isolate test users and correlate durable impression rows. Use
`--warmup-requests` for an excluded warm-up phase. Hot warm-up reuses the
measured user under a separate session ID; unique warm-up uses separate users.

## Output
- Results are written to `loadtest/results/httpx-baseline-<mode>.json` unless `--output` is specified.
- The summary includes:
  - success rate
  - average latency
  - p50 / p95 / p99 latency
  - max latency
  - status code distribution
  - successful-response latency and QPS
  - transport-error and HTTP 5xx rates
  - recommendation-cache hit rate
  - durable/unavailable impression tracking coverage

## Baseline Gate
- `success_rate >= 0.99`
- `p95_ms <= 1000`
- `p99_ms <= 2000`
- no unexpected `5xx` bursts

## Served-Impression Outbox A/B

`scripts/run_recommendation_outbox_ab.py` compares the same checkout with
`RECOMMENDATION_IMPRESSION_LOGGING_ENABLED=false` and `true`. It performs three
paired hot and unique runs in the fixed order off/on, on/off, off/on. Each run
uses a distinct session ID, waits up to 60 seconds for the measured outbox rows
to drain, and reconciles published rows against durable impression responses.

Prepare a dedicated Compose project and its sample index for a directional
local run:

```bash
export COMPOSE_PROJECT_NAME="vc-outbox-ab-$(date +%Y%m%d%H%M%S)"
export ENVIRONMENT=test
export VECTOR_BOOTSTRAP_MODE=sample
export SERVICE_GATEWAY_WORKERS=1
export SERVICE_RECOMMENDATION_WORKERS=1
export SERVICE_RANKING_WORKERS=1
export RANKING_RUNNER_REPLICAS=1

docker compose --profile demo build vector-sample-bootstrap
docker compose --profile demo run --rm vector-sample-bootstrap
docker compose up -d --build caddy

python scripts/run_recommendation_outbox_ab.py \
  --base-url http://127.0.0.1 \
  --compose-project-name "$COMPOSE_PROJECT_NAME" \
  --requests 3000 \
  --concurrency 100 \
  --warmup-requests 300 \
  --environment-class directional

docker compose -p "$COMPOSE_PROJECT_NAME" down -v --remove-orphans
```

For a release go/no-go run, use fixed Linux resources, identical verified
model/vector artifacts, `--environment-class fixed-linux`, and
`--artifact-manifest <path>`. The runner records the commit, host information,
artifact-manifest SHA-256, raw runs, medians, regressions, and gate reasons in
`loadtest/results/recommendation-outbox-ab/summary.json`. Results remain ignored
artifacts; the summary also captures host CPU/memory and recommendation-container
CPU/memory limits. Record the reviewed summary in `PROGRESS.md`.

Every outbox-enabled run must have error rate below 0.5%, HTTP 5xx below 0.1%,
successful p95 below 1,000 ms, successful p99 below 2,000 ms, durable tracking
coverage of at least 99.5%, zero pending measured rows, and exact durable versus
published reconciliation. Across three runs, median p95 and successful QPS may
not regress by more than 10% versus the disabled control. OrbStack results are
directional and do not establish production capacity.

## Ranking High-Concurrency Acceptance

Ranking acceptance is valid only when runner `/readyz` payloads report
`is_trained=true`, `artifact_verified=true`, the expected model/schema versions,
and `ranking_untrained_fallback_total` remains zero. Record the git commit,
checkpoint version and SHA-256, feature schema, 20-candidate payload, pod CPU and
memory, batch fill, queue wait, runner-slot wait, model-forward latency, and
p50/p95/p99 with every result.

Run the fixed four-runner matrix against the internal ranking HTTP endpoint:

```bash
BASE_URL=http://127.0.0.1:8003 RATE=1500 DURATION=30m MODE=acceptance \
  k6 run --summary-export loadtest/results/ranking-1500qps-30m.json loadtest/k6/ranking.js

BASE_URL=http://127.0.0.1:8003 RATE=2000 DURATION=5m MODE=acceptance \
  k6 run --summary-export loadtest/results/ranking-2000qps-5m.json loadtest/k6/ranking.js

BASE_URL=http://127.0.0.1:8003 RATE=2500 DURATION=5m MODE=overload \
  k6 run --summary-export loadtest/results/ranking-2500qps-overload-5m.json loadtest/k6/ranking.js
```

Acceptance gates are error rate `<0.5%`, 5xx `<0.1%`, successful-response p95
`<400ms`, and p99 `<600ms`. At 2000 QPS the coordinator average CPU must stay
below 85%. In overload mode, requests beyond capacity must return 429 with
`Retry-After: 1`; pool-timeout cascades or 5xx above 0.1% fail the run.

During a separate 1500 QPS run, restart one runner at a time and confirm zero
5xx, p95 below 400ms, runner draining before termination, and no retry after a
batch frame was sent. Finally run a gateway-to-recommendation smoke and verify
the same 429 JSON shape, request ID, and `Retry-After` header at the public edge.

Do not publish ranking-only results as end-to-end recommendation QPS. If no real
trained and verified checkpoint is available, record the load acceptance as
blocked instead of substituting the untrained fallback or a synthetic model.

## Full Recommendation-Path Capacity Test

Use `loadtest/k6/recommendations.js` when the objective is the public synchronous
path rather than the isolated ranker. It exercises Caddy, gateway,
recommendation serving, candidate-cache lookup, ranking coordination, runner
dispatch, and a real Torch forward pass.

A deterministic synthetic checkpoint may be used to find infrastructure
capacity before a trained artifact exists. It must be stored through the normal
artifact manager and reported as `serving_capacity_only`; never use its scores
for relevance or business-quality claims.

Prepare the checkpoint/catalog from a backend container before starting the
runners:

```bash
python scripts/prepare_recommendation_loadtest.py \
  --candidates 200 \
  --seed 20260810 \
  --model-version synthetic-capacity-20260810 \
  --output loadtest/results/recommendation-path/checkpoint-manifest.json
```

Prewarm stable VU user IDs at low concurrency, then use a new `RUN_OFFSET` so
the measured run cannot hit the final recommendation cache:

```bash
MODE=prewarm WARM_USERS=600 PREWARM_VUS=5 \
  BASE_URL=http://127.0.0.1 API_API_KEY="$API_API_KEY" \
  k6 run --summary-export loadtest/results/recommendation-path/prewarm.json \
  loadtest/k6/recommendations.js

MODE=candidate_cache_rank RATE=250 DURATION=30s \
  PRE_ALLOCATED_VUS=300 MAX_VUS=600 WARM_USERS=600 RUN_OFFSET=100000 \
  BASE_URL=http://127.0.0.1 API_API_KEY="$API_API_KEY" \
  k6 run --summary-export loadtest/results/recommendation-path/250qps.json \
  loadtest/k6/recommendations.js
```

The measured gate requires errors below 0.5%, 5xx below 0.1%, no dropped
iterations, successful p95 below 500 ms, p99 below 750 ms, more than 99%
candidate-cache-to-rank responses, more than 99% model-forward evidence, zero
untrained fallback, and fewer than 0.1% final-cache responses. Stop the ramp at
the first failed gate; do not continue to 500/750/1,000 QPS or hide overload by
increasing queue wait.

Generate comparable JSON and Markdown summaries with
`scripts/summarize_recommendation_loadtest.py`. Record runner count, worker
counts, Torch threads, batch/queue settings, load-generator location, host CPU
and memory, checkpoint lineage, candidate count, cache-path rates, batch fill,
model-forward latency, offered and successful QPS, 429/5xx, dropped iterations,
and successful-response p50/p95/p99.

The 2026-08-10 local OrbStack experiment and its limitations are recorded in
`loadtest/results/recommendation-path-20260810/README.md`.

## Ranking ONNX Runtime and Triton A/B

Triton remains opt-in. Prepare a deterministic DCN checkpoint and its ONNX
artifact from the same weights. This runs real forward passes but represents
serving capacity only:

```bash
python scripts/prepare_recommendation_loadtest.py \
  --candidates 200 \
  --seed 20260812 \
  --model-version synthetic-triton-capacity-20260812 \
  --export-onnx \
  --output loadtest/results/ranking-triton-ab/checkpoint-manifest.json
```

Start the opt-in profile with an exact version and no direct coordinator path:

```bash
RANKING_INFERENCE_BACKEND=triton \
RANKING_REQUIRED_MODEL_VERSION=synthetic-triton-capacity-20260812 \
SERVICE_RANKING_COORDINATOR_DIRECT_ENABLED=false \
docker compose --profile triton up -d --build ranking-triton ranking-service
```

The materializer verifies source/ONNX/DIN checksums and lineage before Triton
starts. Confirm `/readyz` reports both the business and numeric model versions,
`triton_server_ready=true`, and `triton_model_ready=true`. Confirm Triton's
request counters increase during every measured case; a final-response cache
hit is not valid forward evidence.

Run legacy and Triton in isolated deployments with the same commit, checkpoint,
CPU quota, two ranking-service workers, 20-candidate payload, and load-generator
location. The helper interleaves A/B order and writes one raw k6 summary per run:

```bash
python scripts/run_ranking_triton_ab.py \
  --legacy-url http://legacy-ranking:8003 \
  --triton-url http://triton-ranking:8003 \
  --rates 500 1000 1500 2000 \
  --duration 5m \
  --repetitions 3
```

Tune one dimension at a time and hold each winner fixed: queue delay
`0/1000/2000/4000us`, max batch `64/128/256`, ORT intra-op threads `1/2/4`,
then instance count `1/2/4`. The corresponding environment settings are
`RANKING_TRITON_QUEUE_DELAY_MICROSECONDS`,
`RANKING_TRITON_MAX_BATCH_SIZE`, `RANKING_TRITON_ORT_INTRA_OP_THREADS`, and
`RANKING_TRITON_INSTANCE_COUNT`. Recreate the immutable Triton pod after every
configuration change. Do not set preferred batch sizes in this phase.

Base and DIN are separate gates. Maximum sustained successful QPS must improve
by at least 10%, p95 may not regress by more than 10%, 5xx must remain below
0.1%, fallback must remain zero, and ranking parity must hold. Run the winner
for 30 minutes, then one higher overload step. Repeat the full path at
`50/100/150/250 QPS` with fixed VU users and forward-counter evidence. If only
ranking-only passes, keep the code opt-in and do not enable staging.

OrbStack can provide a directional smoke only. The go/no-go result must come
from dedicated Linux CPU; never compare an OrbStack run against an older
dedicated-Linux or legacy artifact.
