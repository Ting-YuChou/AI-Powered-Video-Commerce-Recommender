# Event And Artifact Reliability

Apply `migrations/postgres/008_recommendation_event_reliability.sql` before
deploying the updated interaction, recommendation, or Flink services. The
migration is additive and introduces the interaction idempotency ledger,
served-impression outbox, and viewed-impression table.

Apply `migrations/postgres/009_content_task_reliability.sql` before deploying
the gateway, content-task publisher, or content workers. Uploads first persist
the object and then atomically create the authoritative `content_jobs` row and
`content_task_outbox` event. The event ID is UUIDv5 over content ID and content
pipeline version, so every inline or background publish attempt uses the same
Kafka identity. The gateway returns `queued` only after Kafka acknowledges the
event. If publication is still pending, it returns HTTP 503 with
`CONTENT_ENQUEUE_PENDING`, `content_id`, `status=pending_publish`, and
`Retry-After`; the durable object and outbox row remain available for retry.

Run `content-task-publisher` in every environment that accepts uploads. Multiple
replicas safely claim rows with `FOR UPDATE SKIP LOCKED` and owner-token leases.
It also reconciles managed upload objects: unreferenced objects are retained for
the configured grace period before deletion, referenced missing objects
terminalize the job as `failed_missing_object`, and published outbox records are
retained for seven days by default. `content_jobs` is authoritative for status;
Redis is only the completed projection/cache.

Content workers claim `(content_id, pipeline_version)` in
`content_processing_runs`. A valid competing lease skips duplicate model work,
an expired or failed lease can be taken over, and an old owner cannot complete a
new owner's run. Feature artifacts are immutable and committed with the run and
job completion in one database transaction before the Redis projection is
updated. A redelivery after that commit repairs projections from the durable
artifact without recomputing the model. Keep
`MODEL_CONTENT_PIPELINE_VERSION` identical on the gateway and workers; an
explicit mismatched event is failed for investigation. Legacy events without a
version map to the configured worker version during migration.

### Content pipeline failure handling (migration 013)

Apply `migrations/postgres/013_content_upload_idempotency.sql` before deploying
the gateway, content-task publisher, or content workers from Stage B. It is
additive: `content_jobs.idempotency_key`, `content_jobs.request_fingerprint`, a
unique index on `(COALESCE(user_id, ''), idempotency_key)`, an index on
`content_jobs.storage_path`, and `content_processing_runs.redrive_count`.
`auto_create_schema` applies the same statements idempotently at startup.

- **Upload retries.** Clients may send `Idempotency-Key` (1-128 characters of
  `A-Z a-z 0-9 _ . : ~ -`) on `POST /api/content/upload`; a key requires a
  `user_id`. The gateway hashes the body while streaming it and stores a `v1:`
  fingerprint over the body SHA-256, size, filename, and priority.
  - Same key and same request: the original `content_id` and current status
    with `idempotent_replay: true` (a `pending_publish` job is re-published
    inline).
  - Same key, different request: `409 IDEMPOTENCY_KEY_REUSED`.
  - Same key after the job terminally failed: `409 IDEMPOTENCY_KEY_JOB_FAILED`;
    re-upload with a new key.
  - Scope is `(user_id, key)`. With bearer auth `user_id` is bound to the token
    subject, so keys are per authenticated user. In `disabled` or `api_key`
    mode `user_id` is supplied by the trusted caller (the BFF), and two callers
    that send the same `user_id` share one key namespace.
- **Unknown staging outcome.** If staging the job raises (for example the
  connection drops during COMMIT), the gateway looks the job up before touching
  the object: it deletes the object only when the job is definitely absent, keeps
  it and returns `CONTENT_ENQUEUE_PENDING` when the job exists, and keeps it when
  the lookup also fails. The orphan reconciler removes unreferenced objects after
  `SERVICE_CONTENT_ORPHAN_GRACE_SECONDS`.
- **Publisher tasks.** Publishing runs in its own loop. Outbox stats (5 s),
  the reaper, orphan reconciliation (5 min), and pruning (1 h) each run as a
  separate task with their own error guard, so a slow S3 listing or a failing
  maintenance query never delays publishing. Failures are counted in
  `video_commerce_content_task_publisher_errors_total{stage}`.
- **Orphan reconciliation.** The publisher lists upload objects one page at a
  time (`SERVICE_CONTENT_RECONCILE_PAGE_SIZE`, default 1000) and checks each page
  against `content_jobs.storage_path` with one batched `IN` query. Only objects
  directly under the upload prefix are considered; `artifacts/` and
  `content-features/` under the same S3 prefix are never treated as orphans.
- **Dead workers.** A delivery that finds a live competing lease (`lease_held`)
  commits its offset. The publisher's reaper (every
  `SERVICE_CONTENT_PROCESSING_REAPER_INTERVAL_SECONDS`, default 15) finds
  `processing` runs whose lease expired more than
  `SERVICE_CONTENT_PROCESSING_REDRIVE_GRACE_SECONDS` (default 60) ago, using the
  database clock (every lease timestamp is written with the database clock too,
  so app/DB skew cannot expire a live lease), and resets their existing outbox
  row so the same event ID is
  republished; the next worker takes over the expired lease. After
  `SERVICE_CONTENT_PROCESSING_MAX_REDRIVES` (default 3) the run, job, and outbox
  row become terminal `failed` with `processing_attempts_exhausted`. Watch
  `video_commerce_content_processing_redrives_total{outcome}` (`redriven`,
  `exhausted`, `superseded`).
- **Retry classification.** Consumer handler retries use exponential backoff
  with equal jitter (`KAFKA_CONSUMER_HANDLER_RETRY_BACKOFF_MS` base, capped at
  `KAFKA_CONSUMER_HANDLER_RETRY_MAX_BACKOFF_MS`). A `NonRetryableProcessingError`
  (pipeline version mismatch, missing source object, undecodable video) goes to
  the DLQ without retries. A local source file that is not visible yet, I/O
  errors, and model failures stay retryable. All attempts of one message stop
  before 90% of `KAFKA_CONSUMER_MAX_POLL_INTERVAL_MS` and the message is
  dead-lettered, so retries never trigger a consumer rebalance.
- **Lease renewal.** A failed renewal is retried while the current lease is
  still valid; the worker reports `lease_lost` only when the lease was taken or
  is about to expire.
- **Modalities.** A CLIP failure fails the attempt (retryable) instead of
  storing a zero vector as completed. OCR and object-detection failures are
  recorded in `ContentFeatures.modality_status` (`failed`/`degraded`) and the
  job still completes. Watch `video_commerce_content_modality_status_total` and
  `video_commerce_content_zero_norm_embeddings_total`.
- **Local sources.** In `OBJECT_STORAGE_BACKEND=local` mode the upload path is
  the durable source and the worker never deletes it; only temporary files
  materialized from S3 are removed.

`video-processing-tasks` defaults to 12 partitions in Compose and Helm so up to
12 content workers can consume in parallel. The Compose `kafka-init` service and
the Helm topic init job (`alterPartitions: true`) also run `kafka-topics --alter`
for this topic, which grows an existing 3-partition topic and is a no-op
otherwise. Keys are re-hashed after a partition increase, and partitions can
never be decreased. To do it by hand, during a quiet period:

```bash
kafka-topics --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --alter \
  --topic video-processing-tasks --partitions 12
kafka-topics --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --describe \
  --topic video-processing-tasks
```

In Compose, run the same command with
`docker compose exec kafka kafka-topics --bootstrap-server kafka:9092 ...`.

Production must set `VECTOR_BOOTSTRAP_MODE=required`. A vector activation is
accepted only when the FAISS file, metadata, optional embedding sidecar, and
manifest agree on checksums, dimensions, row count, and index map. Local demos
may set `ENVIRONMENT=development` and `VECTOR_BOOTSTRAP_MODE=sample`; the
`vector-sample-bootstrap` one-shot service creates that artifact before the
application starts. `load_index()` never generates sample products.
Local/shared-volume publication writes an immutable directory under
`<index>.generations/` and changes only `<index>.active.json` after the whole
generation has been verified. When
`RETRIEVAL_VISUAL_PRODUCT_INDEX_MANIFEST_PATH` is configured, recommendation
startup first downloads the recorded object-storage bundle to staging, verifies
every checksum and lineage field, and activates it directly without requiring a
legacy local index file.

`./startup.sh start` enables the Flink profile and submits
`video-commerce-interaction-features` idempotently. `./startup.sh health`
requires exactly one running job. The `FlinkInteractionFeatureJobMissing`
alert detects a missing or duplicated job from JobManager metrics.

Interaction clients should create one UUID per user action and reuse it for
HTTP retries. The ingest service leases that ID in Postgres, rejects payload
collisions with HTTP 409, and marks it published only after Kafka acknowledges
the write. Lease completion is conditional on its owner token, so an expired
publisher cannot overwrite a newer claimant. Flink and Postgres then
deduplicate on the same event ID.

Recommendation responses expose an impression ID only after the complete
served slate is in the Postgres outbox. Clients report an item as viewed after
at least 50 percent remains in the viewport for one second. Unknown impression
IDs, products, positions, or users outside the durable slate are isolated and
counted. Training negatives are drawn from viewed items inside the configured
attribution window;
legacy served-only rows remain useful for delivery-funnel analysis only.

Public event times come from client clocks, but the PIT jobs anchor on
`event_time`, so ingest bounds them with server times. A view's `event_time`
is the client `viewed_at` clamped into `[served_at, server_received_at]`, where
`served_at` is the durable slate's `created_at`. An interaction's `event_time`
is clamped to at most `server_received_at`; its lower bound is the PIT jobs'
server `available_at`, not an ingest lookup. Events keep the raw client value
as `client_viewed_at` or `client_event_time` (null when the server filled the
time). Interaction idempotency hashes the client value, so retries stay
duplicates even when each retry clamps differently. Watch
`video_commerce_public_event_clock_skew_seconds{kind}` (client minus receive
time; signed buckets, so no `_sum`) and
`video_commerce_public_event_time_clamped_total{kind,bound}`.

The ranking artifact records `score_policy_version` and value-normalization
metadata. `business-value-v1` computes the canonical serving score as clipped
CTCVR multiplied by inverse-transformed predicted value; disabling business
scoring uses the raw ranking score. Training objectives, validation NDCG, and
serving post-processing share the same Torch/NumPy policy and product-ID tie
breaker. Keep the previous verified artifact available until shadow parity has
passed for Python and exported inference. Both latest and exact-version
production activation reject missing policy or value-normalization metadata.

Use `video_commerce_recommendation_impression_events_total` by stage and
status together with Kafka consumer lag, Flink job state, and the durable
outbox retry fields for reconciliation. Exclude responses with
`impression_tracking=unavailable` from attribution and training datasets.
For content delivery, alert on `content_task_outbox_pending`,
`content_task_outbox_oldest_age_seconds`, missing-object retries, and
`content_processing_events_total{outcome="lease_lost"}`. A growing outbox means
uploads may be durable but are not yet available to workers.

For an isolated release smoke with real Kafka and Postgres, use a disposable
Compose project and test-only vector mode:

```bash
ENVIRONMENT=test VECTOR_BOOTSTRAP_MODE=empty \
  docker compose -p vc-phase-a-smoke --profile test up -d \
  postgres redis redis-cache zookeeper kafka kafka-init
ENVIRONMENT=test VECTOR_BOOTSTRAP_MODE=empty \
  docker compose -p vc-phase-a-smoke run --rm --build \
  content-task-publisher python -m scripts.content_reliability_smoke
docker compose -p vc-phase-a-smoke --profile test down -v
```

The script verifies ordinary publication, Kafka-ack-before-outbox-mark
redelivery with a stable event ID, one model computation across worker
redelivery, and durable projection repair. The project name and volumes are
separate from the operator's normal stack.

The PR-required Flink closed-loop smoke uses the same isolation rule:

```bash
COMPOSE_PROJECT_NAME=vc-flink-closed-loop-local \
  scripts/run_flink_closed_loop_smoke.sh
```

It bootstraps a test-only sample vector generation, starts exactly one official
`video-commerce-interaction-features` job, and sends one stable click event. The
test waits for one Postgres interaction row and the matching official Redis
`uf:`/`uiz:` state, retries the event to prove no duplicate effects, then asks
for a recommendation. The final assertion reads the durable served-impression
outbox and requires its `user_feature_snapshot` to contain the Flink-produced
interaction. A failure saves Compose logs and Flink REST state under
`FLINK_CLOSED_LOOP_ARTIFACT_DIR`; the runner always removes only its named
project and volumes.

The runner disables the known-user negative snapshot optimization so a newly
created test user immediately exercises the authoritative official Redis keys.
Production keeps the optimization enabled; its refresh interval can delay the
first feature read for a newly observed user and is tracked as a separate
freshness concern.
