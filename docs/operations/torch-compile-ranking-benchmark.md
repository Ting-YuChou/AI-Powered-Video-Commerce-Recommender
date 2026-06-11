# Ranking torch.compile Benchmark Notes

Use this flow to validate ranking `torch.compile` locally without making GPU
performance claims. Local CPU results are correctness and regression signals
only.

## CPU Smoke

Run the model-level parity and health smoke from the repository root:

```bash
python scripts/torch_compile_ranking_cpu_smoke.py --backend inductor --mode default --batch-size 32 --iterations 3
```

Expected result:

- `status` is `passed`.
- `parity.passed` is `true`.
- `compiled_health.torch_compile_active` is `true`.
- `compiled_health.torch_compile_fallback_count` is `0`.
- `compiled_health.torch_compile_error` is `null`.

The timing fields in this script are CPU regression hints only. Do not use them
to claim GPU kernel-fusion latency or throughput wins.

## Local Load Baseline Toggle

Use the same request shape for eager and compiled runs. Restart the ranking
runtime between modes so the compile setting is applied at startup.

Eager baseline:

```bash
RANKING_TORCH_COMPILE_ENABLED=false \
MONITORING_ENABLE_PROFILING_LOGS=true \
docker compose up -d ranking-coordinator ranking-runner ranking-service recommendation-service gateway-api caddy

python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode hot --timeout 10 --output /tmp/ranking-eager-hot.json
python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode unique --timeout 10 --output /tmp/ranking-eager-unique.json
```

Compiled smoke:

```bash
RANKING_TORCH_COMPILE_ENABLED=true \
RANKING_TORCH_COMPILE_BACKEND=inductor \
RANKING_TORCH_COMPILE_MODE=default \
RANKING_TORCH_COMPILE_DYNAMIC=true \
MONITORING_ENABLE_PROFILING_LOGS=true \
docker compose up -d ranking-coordinator ranking-runner ranking-service recommendation-service gateway-api caddy

python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode hot --timeout 10 --output /tmp/ranking-compile-hot.json
python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 3000 --concurrency 100 --mode unique --timeout 10 --output /tmp/ranking-compile-unique.json
```

For local CPU-only machines, treat the load results as startup/fallback/error
regression checks. A production canary still requires a GPU runtime with
`MODEL_DEVICE=cuda`.

## Evidence To Record

- Load script output: `p50_ms`, `p95_ms`, `p99_ms`, `qps_total`,
  `success_rate`, and `server_error_count`.
- Ranking health/status: `torch_compile_active`, `torch_compile_warmup_ms`,
  `torch_compile_fallback_count`, `torch_compile_last_fallback_error`, and
  `torch_compile_last_inference_path`.
- Ranking profile logs: `tensor_prep_ms`, `model_forward_ms`, `total_ms`, and
  `inference_path`.
- On GPU hosts only: GPU memory, utilization, startup compile time, and error
  rate for enabled/disabled runs.
