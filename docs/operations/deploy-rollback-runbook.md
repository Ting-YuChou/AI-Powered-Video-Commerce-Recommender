# Deploy And Rollback Runbook

See [Event and artifact reliability](event-and-artifact-reliability.md) for
the migration order, vector activation gate, durable impression flow, and
score-policy rollback contract.

## Deploy
1. Apply additive migrations in order through
   `migrations/postgres/009_content_task_reliability.sql`. The gateway must not
   accept production uploads until the content outbox and processing-run tables
   exist.
2. Build and verify images:
   - `docker compose config -q`
   - `docker compose --profile test run --rm backend-tests`
3. Apply the stack:
   - `./startup.sh start` (includes the authoritative Flink profile and the
     explicit local sample-index bootstrap when configured)
4. Verify health:
   - `docker compose ps`
   - `curl http://localhost/`
   - `curl http://localhost:8000/readyz` from inside the network or via `docker compose exec gateway-api`
   - verify `content-task-publisher` metrics are scrapeable and both content
     outbox backlog gauges drain to zero
5. Run the smoke baseline:
   - `python scripts/loadtest_api_baseline.py --base-url http://127.0.0.1 --requests 500 --concurrency 50 --mode hot`

## Rollback
1. Identify the last known-good image tags or Git revision.
2. Rebuild or retag the last known-good revision.
3. Re-apply:
   - `docker compose --profile flink up -d --build`
4. Re-run readiness and smoke checks before reopening traffic.

The content migration is additive and should remain in place during rollback.
If rolling back the publisher or worker, first stop new uploads, wait for
in-flight processing leases to expire or complete, and record the remaining
`pending_publish` rows. Do not delete their objects or outbox rows. A compatible
forward deployment can resume them with the same event IDs. Older workers can
read legacy events, but they do not provide pipeline-version lease fencing, so
do not run old and new content workers concurrently.

## Release Gates
- `backend-tests` passes.
- `integration-tests` passes before major release or infrastructure change.
- Gateway `/readyz` is healthy.
- `content-task-publisher` is running and content outbox backlog is draining.
- Prometheus shows no active critical alerts after deployment.
