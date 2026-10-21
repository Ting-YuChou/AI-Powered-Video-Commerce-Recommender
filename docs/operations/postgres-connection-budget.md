# Postgres connection budget and timeouts

Every service that opens a `SystemStore` owns one SQLAlchemy pool **per
process**. A uvicorn service with `--workers 4` therefore holds four pools. The
worst-case connection count is:

```
sum over workloads: pods × processes per pod × (DATABASE_POOL_SIZE + DATABASE_MAX_OVERFLOW)
```

Pools open connections lazily, so steady-state usage is lower. The budget must
still hold at the worst case: an HPA scale-out followed by a rolling update is
exactly when Postgres would otherwise answer `too many connections`, and the
ingest ledger and impression outbox would fail together.

## Helm budget (HPA max plus rollout surge)

`tests/test_connection_budget.py` recomputes this table from
`charts/video-commerce/values.yaml` and fails when the total plus
`postgres.reservedConnections` exceeds `postgres.maxConnectionsBudget`. It
counts optional workloads as enabled and adds the Deployment surge (25% unless a
workload sets `strategy.maxSurge`), so enabling a component or raising
`maxReplicas` cannot silently break the budget.

| Workload | Pods (max + surge) | Processes | Conns/process | Total |
| --- | ---: | ---: | ---: | ---: |
| gateway-api | 8 + 2 | 4 | 2 + 1 | 120 |
| recommendation-service | 10 + 3 | 4 | 2 + 1 | 156 |
| ranking-service | 8 + 2 | 2 | 1 + 0 | 20 |
| ranking-coordinator | 1 + 1 | 1 | 1 + 0 | 2 |
| ranking-runner | 32 + 1 | 1 | 1 + 0 | 33 |
| interaction-ingest-service | 8 + 2 | 2 | 2 + 1 | 60 |
| content-worker | 1 + 1 | 1 | 2 + 1 | 6 |
| content-task-publisher | 2 + 1 | 1 | 2 + 1 | 9 |
| feature-worker (optional) | 2 + 1 | 1 | 2 + 1 | 9 |
| catalog-event-publisher (optional) | 2 + 1 | 1 | 2 + 1 | 9 |
| pit-state-exporter (optional) | 1 + 1 | 1 | 2 + 1 | 6 |
| model-trainer | 1 + 1 | 1 | 3 + 2 | 10 |
| pit-orchestrator CronJob | 1 | 1 | 2 + 1 | 3 |
| iceberg-maintenance CronJob | 1 | 1 | 2 + 1 | 3 |
| **Total** | | | | **446** |

With `reservedConnections: 25` for superuser, migration, exporter and `psql`
sessions, the default `maxConnectionsBudget: 500` holds. Before the change the
same fleet needed about 1,400 connections (5 + 5 per process everywhere).

Set `postgres.maxConnectionsBudget` to the external server's
`max_connections` minus `superuser_reserved_connections`, or to PgBouncer's
`max_db_connections` when a pooler sits in front. To raise a pool, raise the
budget (and the server) in the same change; the test shows the per-workload
table when it fails.

The ranking tier (`ranking-service`, coordinator, runners) only polls the
active release pointer and writes rollout observations, so it runs with one
connection per process. Hot-path services keep `2 + 1`: their transactions are
short and the pool timeout below bounds the wait when it is exhausted.

## Compose budget

Compose runs one replica of each service (four runners) against the bundled
`postgres:16-alpine`, whose default `max_connections` is 100.

| Service | Processes | Conns/process | Total |
| --- | ---: | ---: | ---: |
| gateway-api | 4 | 2 + 1 | 12 |
| recommendation-service | 8 | 2 + 1 | 24 |
| ranking-service | 2 | 1 + 0 | 2 |
| ranking-coordinator | 1 | 1 + 0 | 1 |
| ranking-runner ×4 | 4 | 1 + 0 | 4 |
| interaction-ingest-service | 2 | 2 + 1 | 6 |
| content-worker, content-task-publisher, feature-worker, catalog-event-publisher | 4 | 2 + 1 | 12 |
| model-trainer | 1 | 3 + 2 | 5 |
| pit-dataset-orchestrator, pit-state-exporter (`pit-production`) | 2 | 2 + 1 | 6 |
| **Total** | | | **72** |

Raising `SERVICE_RECOMMENDATION_WORKERS` or `RANKING_RUNNER_REPLICAS` changes
this table; keep the total below roughly 90 or raise Postgres `max_connections`.

## Timeouts

| Setting | Default | Scope |
| --- | --- | --- |
| `DATABASE_POOL_TIMEOUT_SECONDS` | 2 | Wait for a pooled connection. Offline clients (model-trainer, PIT orchestrator, iceberg maintenance) override it to 30. |
| `DATABASE_CONNECT_TIMEOUT_SECONDS` | 5 | asyncpg connection establishment. |
| `DATABASE_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS` | 60000 | Server-side; kills sessions left idle inside a transaction. |
| `DATABASE_POOL_RECYCLE_SECONDS` | 1800 | Replace pooled connections older than this. |
| `DATABASE_HOT_PATH_STATEMENT_TIMEOUT_MS` | 1000 | `SET LOCAL statement_timeout` inside request-path transactions only. |
| `DATABASE_SCHEMA_LOCK_TIMEOUT_MS` | 5000 | `SET LOCAL lock_timeout` for startup DDL, applied after the schema lock is held. |
| `DATABASE_STATEMENT_CACHE_SIZE` | unset | asyncpg prepared-statement cache; set `0` behind PgBouncer. |

There is deliberately **no global** `statement_timeout`. Training reads, PIT
orchestration and iceberg maintenance run long scans that must not be cut off.
The request-path methods open their transaction through
`SystemStore._hot_path_session()`, which issues `SET LOCAL statement_timeout`:

- the recommendation outbox enqueue,
- the interaction idempotency ledger claim, mark and release,
- the impression slate and known-impression lookups used by view ingest,
- the readiness `SELECT 1`.

`SET LOCAL` ends with the transaction, so it never leaks to the next borrower of
a pooled connection and works under PgBouncer transaction pooling.

### Served-impression budget

The recommendation response waits at most
`RECOMMENDATION_IMPRESSION_OUTBOX_TIMEOUT_MS` (100 ms) for the outbox insert,
including cache hits and singleflight joiners. Over budget, the slate is still
returned, but without `impression_id` (`impression_tracking: unavailable`), and
`video_commerce_recommendation_impression_events_total{stage="served",status="timeout"}`
increments.

Inserts are micro-batched per process. Responses finish in bursts (ranking runs
in batches), and one transaction per impression made each burst compete for the
two or three pooled connections. Before batching, about a third of the
impressions at concurrency 20 were skipped with a healthy database. Now:

- At most `RECOMMENDATION_IMPRESSION_OUTBOX_MAX_INFLIGHT` (2) write
  transactions run per process. Keep it below
  `DATABASE_POOL_SIZE + DATABASE_MAX_OVERFLOW` so the outbox dispatcher keeps a
  connection.
- Impressions that arrive while those are busy wait and commit together in
  the next transaction: one multi-row `INSERT ... ON CONFLICT DO NOTHING
  RETURNING`, up to `RECOMMENDATION_IMPRESSION_OUTBOX_MAX_BATCH_SIZE` (32) rows.
- At most `MAX_INFLIGHT * MAX_BATCH_SIZE` rows wait. Beyond that the impression
  is skipped immediately (`status="unavailable"`) instead of queueing.
- A row still queued when its deadline passes is dropped without a DB write.

`video_commerce_recommendation_impression_untracked_total{reason}` says why a
slate went out without an `impression_id`:

| Reason | Meaning |
| --- | --- |
| `timeout` | The budget ran out (slow write, or queued behind slow writes). |
| `queue_full` | The per-process queue was full; nothing was written. |
| `pool_timeout` | No pooled connection within `DATABASE_POOL_TIMEOUT_SECONDS`. |
| `db_error` | The write failed before COMMIT, for example on the statement timeout. |
| `error` | The write failed in a way that may have committed (see Retire). |

The insert is shielded, not cancelled: cancelling an asyncpg statement inside a
transaction can return a broken connection to the pool. Instead:

1. **Deadline rollback.** Each row carries its serving deadline. Rows whose
   deadline has passed when the INSERT is ready to commit are deleted inside
   the transaction, or the transaction rolls back when that is every row it
   inserted.
2. **Hold-off.** Rows are staged with `next_attempt_at = now + budget + 1 s`, so
   no replica's dispatcher can claim them for that long. Impression events
   therefore reach Kafka about one to two seconds after serving. A local insert
   restarts the dispatcher's short polls.
3. **Retire.** A COMMIT that lands after the deadline, or a write error raised
   by COMMIT itself (a dropped connection can still have committed), is
   retired as terminal `not_exposed`. A write that fails before COMMIT is sent
   (pool timeout, statement timeout, deadline rollback) committed nothing and
   is not retired. Before this distinction, every insert that hit the
   statement timeout behind a table lock queued a retire, which then blocked
   on the same lock and ended as `retire_failed`.
   - A retired row is never published and never resolves a view slate.
   - It counts in
     `video_commerce_recommendation_outbox_terminal_total{reason="not_exposed"}`.
   - The retire is a conditional UPDATE that refuses rows with a live claim.
     It is bounded by the hot-path statement timeout and retried three times
     with backoff; a client-side cancel could abandon the statement mid-write.

Outcomes in `video_commerce_recommendation_impression_events_total{stage="served"}`:

| Status | Meaning |
| --- | --- |
| `not_exposed_published` | A dispatcher still claimed or published the row first, which is only possible when the hold-off was exceeded. The event can reach Kafka with an `impression_id` no client holds. No view can reference it, and the PIT and training paths use viewed rows, so it only appears in served-funnel counts. |
| `retire_failed` | Every retire attempt errored for a write whose COMMIT outcome was unknown. If that row did commit, it publishes normally, with the same served-funnel impact. |

## Startup DDL lock

With `DATABASE_AUTO_CREATE_SCHEMA=true` every service runs `create_all` and the
additive `_ensure_*` steps at startup. They now run in one transaction that
first takes `pg_advisory_xact_lock(1947420118)`. Replicas that start together
queue behind the holder instead of racing on `pg_type` / `pg_class` and failing
with duplicate-key errors. `lock_timeout` is set only after the advisory lock
is held, so a waiting replica keeps waiting while the holder's own DDL cannot
wedge behind a long-running reader. The additive steps check
`information_schema` and `to_regclass` first and only issue `ALTER TABLE` or
`CREATE INDEX` when something is actually missing. Even a no-op
`ADD COLUMN IF NOT EXISTS` takes an `ACCESS EXCLUSIVE` lock, which would stall
the impression outbox on every pod restart.

If the holder's DDL hits `lock_timeout`, that replica fails startup and is
restarted.

**Production: run migrations as a pre-deploy step.** Apply the numbered files
in `migrations/postgres/` (including `012_recommendation_outbox_terminal.sql`,
whose indexes use `CREATE INDEX CONCURRENTLY`; run it with psql autocommit,
not inside a transaction) before rolling out. Then run the services with
`DATABASE_AUTO_CREATE_SCHEMA=false`. Otherwise the first new pod performs the
`ALTER TABLE` on the hot impression outbox during the rolling deploy. That ALTER
holds `ACCESS EXCLUSIVE` and can stall every replica's impression inserts for up
to `DATABASE_SCHEMA_LOCK_TIMEOUT_MS`; the serving budget then turns those into
`timeout` outcomes.

## PgBouncer (documented, not deployed)

The chart does not deploy a pooler. To put PgBouncer in front of Postgres:

- Use `pool_mode = transaction` and point `DATABASE_URL` at PgBouncer.
- Set `external.database.statementCacheSize: "0"` (`DATABASE_STATEMENT_CACHE_SIZE=0`).
  asyncpg otherwise prepares statements on one server connection and reuses
  them on another, which fails with `prepared statement "__asyncpg_stmt_…" does
  not exist`.
- `SET LOCAL statement_timeout`, `SET LOCAL lock_timeout` and
  `pg_advisory_xact_lock` are all transaction-scoped and safe in transaction
  mode. Session-level `SET` and session advisory locks are not used.
- `server_settings` such as `idle_in_transaction_session_timeout` are sent as
  startup parameters. Add them to PgBouncer's `ignore_startup_parameters`, or
  configure them on the server or role instead.
- Size `max_db_connections` from the budget table above. The client-side pools
  can then be raised, because PgBouncer multiplexes them onto that fixed
  server-side budget.
