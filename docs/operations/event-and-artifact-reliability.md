# Event And Artifact Reliability

Apply `migrations/postgres/008_recommendation_event_reliability.sql` before
deploying the updated interaction, recommendation, or Flink services. The
migration is additive and introduces the interaction idempotency ledger,
served-impression outbox, and viewed-impression table.

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
