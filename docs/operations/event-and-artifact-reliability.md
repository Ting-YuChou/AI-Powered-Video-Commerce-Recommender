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
job completion in one database transaction before Redis and vector projections
are updated. A redelivery after that commit repairs projections from the durable
artifact without recomputing the model. Keep
`MODEL_CONTENT_PIPELINE_VERSION` identical on the gateway and workers; an
explicit mismatched event is failed for investigation. Legacy events without a
version map to the configured worker version during migration.

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
