# Deploy And Rollback Runbook

See [Event and artifact reliability](event-and-artifact-reliability.md) for
the migration order, vector activation gate, durable impression flow, and
score-policy rollback contract.

See [Flink realtime feature recovery](flink-realtime-recovery.md) for
interaction-feature savepoints, state epochs, late-event quarantine, and
bounded DLQ replay.

See [Model artifact rolling activation](model-artifact-rolling-activation.md)
for generation convergence and rollback, and
[Iceberg maintenance](iceberg-maintenance.md) for dry-run, execute, and
post-cleanup verification.

## Deploy
1. Apply additive migrations in order through
   `migrations/postgres/011_artifact_rollout_and_iceberg_maintenance.sql`. The gateway must not
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
3. Take a savepoint of the realtime feature job before replacing images:
   - `scripts/flink_realtime_ops.sh stop-with-savepoint` (record the path)
4. Re-apply:
   - `docker compose --profile flink up -d --build`
   - The submitter restores the newest retained checkpoint automatically. If
     the rolled-back jar needs the recorded savepoint instead, run
     `scripts/flink_realtime_ops.sh restore <savepoint-path>`. See
     `docs/operations/flink-realtime-recovery.md`.
5. Re-run readiness and smoke checks before reopening traffic.

The content migration is additive and should remain in place during rollback.
The model release migration is also additive and remains in place. Use the
audited active-pointer rollback in
`docs/operations/model-release-runbook.md`; do not delete release evidence or
manually rewrite pointer rows.
Two-Tower retrieval has a separate family-specific workflow in
`docs/operations/retrieval-pit-release-runbook.md`. Keep its PIT and catalog
artifacts immutable and roll back through the generation-fenced CLI.
For PIT reader rollback, keep the verified manifest and model release fixed,
set `FEATURE_LAKE_PIT_READER_MODE=materialized` only in a non-production shadow
trainer, and compare split IDs, normalization, mapping, loss, and gate evidence.
Production rejects materialized mode; restore the previous trainer image if the
streaming implementation itself must be rolled back.
If rolling back the publisher or worker, first stop new uploads, wait for
in-flight processing leases to expire or complete, and record the remaining
`pending_publish` rows. Do not delete their objects or outbox rows. A compatible
forward deployment can resume them with the same event IDs. Older workers can
read legacy events, but they do not provide pipeline-version lease fencing, so
do not run old and new content workers concurrently.

## Release Gates
- The `docker-backend-tests` CI check passes.
- `integration-tests` passes before major release or infrastructure change.
- Gateway `/readyz` is healthy.
- `content-task-publisher` is running and content outbox backlog is draining.
- Prometheus shows no active critical alerts after deployment.
- A real mature retrieval PIT baseline exists before enabling required
  Two-Tower retrieval; synthetic smoke evidence cannot satisfy this gate.
- Every expected model-serving instance reports the active release,
  generation, and bundle digest before the rollout is complete.
- Iceberg execute mode remains disabled until one week of dry-run plans has
  reconciled with protected release and PIT lineage.
