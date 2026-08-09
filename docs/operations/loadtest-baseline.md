# Loadtest Baseline

## Script
- `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode hot --timeout 10`
- `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode unique --timeout 10`

## Output
- Results are written to `loadtest/results/httpx-baseline-<mode>.json` unless `--output` is specified.
- The summary includes:
  - success rate
  - average latency
  - p50 / p95 / p99 latency
  - max latency
  - status code distribution

## Baseline Gate
- `success_rate >= 0.99`
- `p95_ms <= 1000`
- `p99_ms <= 2000`
- no unexpected `5xx` bursts

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
