# Iceberg maintenance

Maintenance is restricted to the feature-lake allowlist in `video_commerce/ml/iceberg_maintenance.py`. Planning protects the current snapshot, the newest 20 snapshots, active and staging release lineage, recent retired release lineage, and live or rollback-eligible PIT runs. The default retention is 30 days; orphan files must be at least seven days old.

Start with read-only inspection and planning:

```bash
python scripts/iceberg_maintenance.py status
python scripts/iceberg_maintenance.py plan --table ranking_training_pit
```

The plan is persisted in `iceberg_maintenance_runs` with its table head and SHA-256. Execute only the returned run and hash:

```bash
python scripts/iceberg_maintenance.py execute \
  --run-id iceberg-ranking_training_pit-... \
  --plan-sha256 ...
python scripts/iceberg_maintenance.py verify \
  --run-id iceberg-ranking_training_pit-...
```

Execution fails if the table head changed, the plan bytes changed, or a newly protected reference intersects an expiration candidate. Snapshot IDs are passed explicitly to the Flink job. Iceberg `DeleteOrphanFiles` performs prefix validation in `ERROR` mode and applies the seven-day minimum age and deletion batch limit.

The Helm CronJob runs Sunday at 03:30 UTC with `concurrencyPolicy: Forbid`. Keep `external.featureLake.icebergMaintenanceExecute=false` for the first week and inspect every persisted plan. Enable execute only after the table heads, protected snapshot references, candidate counts, and storage prefixes reconcile. A failed run must be investigated before the next execute window; never bypass the plan hash or head-snapshot check.

This maintenance job does not compact or rewrite data files. Verification confirms protected snapshots remain readable and planned expired snapshots are gone. Run an application read and a PIT reader smoke after the first production cleanup before treating the schedule as established.
