# Deploy And Rollback Runbook

## Deploy
1. Build and verify images:
   - `docker compose config -q`
   - `docker compose --profile test run --rm backend-tests`
2. Apply the stack:
   - `docker compose up -d --build`
3. Verify health:
   - `docker compose ps`
   - `curl http://localhost/`
   - `curl http://localhost:8000/readyz` from inside the network or via `docker compose exec gateway-api`
4. Run the smoke baseline:
   - `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 500 --concurrency 50 --mode hot`

## Rollback
1. Identify the last known-good image tags or Git revision.
2. Rebuild or retag the last known-good revision.
3. Re-apply:
   - `docker compose up -d --build`
4. Re-run readiness and smoke checks before reopening traffic.

## Release Gates
- `backend-tests` passes.
- `integration-tests` passes before major release or infrastructure change.
- Gateway `/readyz` is healthy.
- Prometheus shows no active critical alerts after deployment.
