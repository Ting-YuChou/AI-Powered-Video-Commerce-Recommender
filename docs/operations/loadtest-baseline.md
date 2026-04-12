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
