# Alert Runbooks

One section per alert in `monitoring/prometheus-rules.yml`. Every rule's
`runbook_url` annotation links to its section here, and
`tests/test_alert_runbooks.py` fails if an alert has no `runbook_url` or its
anchor is missing. Add a section whenever you add an alert.

Routing (`monitoring/alertmanager.yml`): `critical` pages the `oncall-webhook`
receiver, `warning` goes to `default` (Alertmanager UI/API only). A critical
alert inhibits the warning of the same name, and
`FlinkInteractionFeatureJobMissing` inhibits the checkpoint and watermark
alerts. Receiver setup is in [SLO And Alerting](slo-and-alerting.md).

Where to look:

- Compose: Prometheus `http://127.0.0.1:9090`, Alertmanager
  `http://127.0.0.1:9093`, Grafana `http://127.0.0.1:3000`, Jaeger
  `http://127.0.0.1:16686`; `docker compose ps` and
  `docker compose logs --since 30m <service>`.
- Kubernetes: the same services' pod logs and events in the release
  namespace. Rules are installed through the chart's optional PrometheusRule.
- `/readyz` is the shallow readiness probe; `/readyz/full` (authenticated)
  reports every dependency and worker heartbeat.
- Multi-worker services aggregate metrics across uvicorn workers
  (`PROMETHEUS_MULTIPROC_DIR`); see the limits in
  [Observability Validation](observability-validation.md#multi-worker-metrics).

To silence a known issue while you work on it:

```bash
docker compose exec alertmanager amtool silence add alertname=<AlertName> \
  --alertmanager.url=http://127.0.0.1:9093 --duration=2h --comment="<ticket>"
```

## ServiceErrorBudgetBurn

- **Symptom:** more than 5% of a job's HTTP responses are 5xx for 10 minutes.
- **Impact:** user-facing failures; the 30-day availability SLO burns fast.
- **First checks:** which `path` and `status` dominate
  (`sum by (path, status) (rate(video_commerce_http_requests_total{job="<job>",status=~"5.."}[5m]))`);
  service logs for `request_failed`, then the `trace_id` in Jaeger; the
  service's `/readyz/full`.
- **Remediation:** a 503 from a dependency check means fix that dependency
  (see its alert). A new release: roll back per
  [Deploy And Rollback](deploy-rollback-runbook.md#rollback). Capacity: scale
  replicas, since overload should return 429 rather than 5xx.

## ServiceP95LatencyHigh

- **Symptom:** a job's p95 latency is above 1 s for 10 minutes.
- **Impact:** slow pages and API clients; precedes timeouts and 5xx.
- **First checks:** the slowest `path` from
  `video_commerce_http_request_duration_seconds_bucket`; database p95
  (`DatabaseP95LatencyHigh`), ranking queue waits, Redis latency; CPU
  saturation (`video_commerce_process_cpu_percent`).
- **Remediation:** remove the bottleneck dependency first. Scale the service
  or its dependency. Record a load baseline before claiming a fix
  ([Loadtest Baseline](loadtest-baseline.md)).

## ServiceP99LatencyHigh

- **Symptom:** a job's p99 latency is above 2.5 s for 10 minutes.
- **Impact:** tail requests hit client and gateway timeouts.
- **First checks:** as for [ServiceP95LatencyHigh](#servicep95latencyhigh);
  tail-only regressions usually come from lock or pool waits
  (`video_commerce_database_pool_connections`), GC, or ranking batch
  queueing (`RankingQueueWaitP95High`).
- **Remediation:** fix the queueing resource; check pool sizes against the
  connection budget before raising them.

## GatewayUnavailable

- **Symptom:** Prometheus cannot scrape `gateway-api` for 2 minutes.
- **Impact:** the whole public API is down (Caddy only serves static pages).
- **First checks:** `docker compose ps gateway-api` / pod status and restarts;
  container logs for startup failures (config validation, auth not
  configured); OOM kills.
- **Remediation:** fix the startup error and restart. If a release caused it,
  roll back ([Deploy And Rollback](deploy-rollback-runbook.md#rollback)).

## RecommendationServiceUnavailable

- **Symptom:** the `recommendation-service` scrape target is down.
- **Impact:** recommendation requests fail at the gateway.
- **First checks:** container state and logs; vector index and model artifact
  load errors at startup; memory limits.
- **Remediation:** restore the failed artifact or dependency and restart; roll
  back the model release if a new generation fails to load
  ([Ranking model release](model-release-runbook.md#rollback)).

## InteractionIngestServiceUnavailable

- **Symptom:** the `interaction-ingest-service` scrape target is down.
- **Impact:** interaction writes fail; features and attribution stop updating.
- **First checks:** container state and logs; Kafka and Postgres reachability
  (ingest needs both to accept an event).
- **Remediation:** fix the dependency and restart. Clients retry with the same
  event ID, so no deduplication cleanup is needed.

## WorkerHeartbeatMissing

- **Symptom:** the gateway sees zero live instances of a background worker for
  2 minutes. `feature-worker` only counts when the Python feature pipeline is
  active (`video_commerce_feature_pipeline_mode{mode!="flink"} == 1`).
- **Impact:** content processing, feature updates or training stop.
- **First checks:** the worker container and logs (`service` label); Redis
  reachability (heartbeats are Redis keys); `/readyz/full` on the gateway.
- **Remediation:** restart the worker after fixing its dependency. In Flink
  mode `feature-worker` is not deployed (Compose profile `python-features`).

## WorkerMetricsTargetDown

- **Symptom:** Prometheus cannot scrape a worker's metrics port for 2 minutes.
  `feature-worker` is only evaluated in Python feature-pipeline mode.
- **Impact:** that worker is down or wedged; its own alerts go blind.
- **First checks:** `docker compose ps <job>`; logs; whether
  `MONITORING_WORKER_METRICS_PORT` changed.
- **Remediation:** restart the worker; if it crash-loops, fix the cause from
  its logs. Start `feature-worker` only with
  `docker compose --profile python-features up -d feature-worker`.

## WorkerProcessingFailures

- **Symptom:** a worker reports message-processing errors for 5 minutes.
- **Impact:** messages are retried and may land in the DLQ.
- **First checks:** worker logs for the exception; `KafkaDeadLetterTraffic`;
  the dependency the handler calls (Postgres, Redis, object storage).
- **Remediation:** fix the dependency or roll back the worker; replay the DLQ
  afterwards ([DLQ replay](flink-realtime-recovery.md#dlq-replay)).

## KafkaProducerDisconnected

- **Symptom:** a producer-owning service (`gateway-api`,
  `interaction-ingest-service`, `recommendation-service`) has had no connected
  Kafka producer for 5 minutes. With several uvicorn workers the value is the
  maximum, so this fires when no worker is connected.
- **Impact:** uploads and interactions return 503; impression events back up in
  the outbox.
- **First checks:** broker health (`docker compose ps kafka`, kafka-exporter
  `up`); service logs for producer start/reconnect errors; topic existence.
- **Remediation:** restore the broker; producers reconnect in the background
  with backoff. Restart the service only if no reconnect is logged after the
  broker is healthy.

## RateLimiterFailingOpen

- **Symptom:** rate-limit checks against Redis error or time out
  (`video_commerce_rate_limiter_errors_total`) for 10 minutes.
- **Impact:** requests are admitted without a limit (fail-open by design), so
  abusive clients are not throttled.
- **First checks:** state Redis latency and errors; `reason` label (timeout vs
  error); `API_RATE_LIMIT_TIMEOUT_MS`.
- **Remediation:** restore Redis health. If abuse is ongoing, block it at the
  edge until the limiter recovers.

## KafkaConsumerLagHigh

- **Symptom:** a consumer group is more than 1000 messages behind for 10
  minutes.
- **Impact:** features, content processing or attribution are stale.
- **First checks:** the group's worker heartbeats and errors; partition
  assignment; processing latency
  (`video_commerce_worker_message_processing_seconds`).
- **Remediation:** fix failing handlers first, then scale consumers up to the
  partition count.

## KafkaDeadLetterTraffic

- **Symptom:** messages are being published to a dead-letter topic.
- **Impact:** those events are not applied until replayed.
- **First checks:** the source topic in `video_commerce_kafka_dead_letter_messages_total`;
  worker logs for the failure class.
- **Remediation:** fix the cause, then replay with the dry-run-first tool
  ([DLQ replay](flink-realtime-recovery.md#dlq-replay)).

## RedisMemoryHigh

- **Symptom:** a Redis role uses more than 80% of `maxmemory` for 10 minutes.
- **Impact:** evictions (cache) or write failures (state) are next.
- **First checks:** which role; key counts by prefix; recent TTL or feature
  namespace changes.
- **Remediation:** raise `maxmemory`, shorten cache TTLs, or remove an
  abandoned feature namespace. Never flush the state instance.

## PostgresExporterDown

- **Symptom:** the Postgres exporter is unreachable or `pg_up == 0`.
- **Impact:** if the database is down, every durable write path fails.
- **First checks:** `docker compose ps postgres postgres-exporter`; database
  logs; connection count against `max_connections`.
- **Remediation:** restore the database; if only the exporter is down, restart
  it.

## DatabaseErrors

- **Symptom:** application Postgres operations fail for 5 minutes.
- **Impact:** depends on the operation; hot-path failures surface as 5xx or
  missing impression IDs.
- **First checks:** `sum by (job, operation) (rate(video_commerce_database_queries_total{status="error"}[5m]))`;
  service logs for the exception; pending migrations.
- **Remediation:** apply missing migrations; fix connectivity or permissions;
  roll back a release that introduced the failing query.

## DatabaseP95LatencyHigh

- **Symptom:** application-observed Postgres p95 is above 500 ms.
- **Impact:** hot paths spend their budget waiting; outbox writes start timing
  out.
- **First checks:** slow queries (`pg_stat_activity`), lock waits, pool
  saturation, background jobs (PIT, Iceberg maintenance, training).
- **Remediation:** stop or reschedule the offending batch job; add the missing
  index; scale the database.

## RankingVerifiedModelMissing

- **Symptom:** no ranking runner reports a verified trained model.
- **Impact:** ranking cannot serve; recommendations fail or fall back.
- **First checks:** runner logs for model, checksum or feature-schema
  readiness failures; the active release pointer
  (`python scripts/model_release.py status`).
- **Remediation:** roll back to the previous verified release
  ([Ranking model release](model-release-runbook.md#rollback)); fix the
  artifact store.

## RankingHealthyRunnersLow

- **Symptom:** fewer than two verified ranking runners are healthy.
- **Impact:** reduced capacity and no redundancy.
- **First checks:** runner pods and drain state
  (`video_commerce_ranking_draining_runners`); runner logs.
- **Remediation:** restore failed runners; undrain runners left draining after
  maintenance (`runner_healthcheck --operation undrain`, loopback only).

## RankingQueueWaitP95High

- **Symptom:** ranking queue wait p95 is above 75 ms.
- **Impact:** ranking latency rises; overload rejections follow.
- **First checks:** request rate vs effective runner capacity
  (`video_commerce_ranking_effective_runner_capacity`); runner health.
- **Remediation:** add runners or restore unhealthy ones.

## RankingRunnerSlotP95High

- **Symptom:** waiting for a runner slot takes more than 50 ms at p95.
- **Impact:** dispatch demand exceeds runner capacity.
- **First checks:** per-endpoint in-flight batches and available connections;
  any runner stuck at its slot limit.
- **Remediation:** scale runners; investigate a slow runner (CPU, model
  generation) before raising slot limits.

## RankingOverloadSustained

- **Symptom:** ranking admission rejects more than one request per second
  (HTTP 429) for 5 minutes.
- **Impact:** some recommendations are served without ranking or fail.
- **First checks:** `reason` on `video_commerce_ranking_admission_rejections_total`;
  traffic changes; runner capacity.
- **Remediation:** scale ranking capacity; shed non-critical traffic.

## RankingCoordinatorPoolTimeout

- **Symptom:** the ranking coordinator client timed out (pool or request).
- **Impact:** ranking requests fail for the affected callers.
- **First checks:** coordinator CPU and connection pool saturation; runner
  dispatch latency; network between services.
- **Remediation:** scale the coordinator's callers or the coordinator; restore
  runner capacity.

## Ranking5xxDetected

- **Symptom:** ranking or recommendation returns 5xx.
- **Impact:** user-visible failures; overload is supposed to return 429.
- **First checks:** logs for model, protocol or dependency errors; whether a
  release just activated.
- **Remediation:** roll back the release or fix the dependency.

## RankingTritonModelUnavailable

- **Symptom:** the exact required Triton model version is not ready.
- **Impact:** Triton-backed ranking cannot serve.
- **First checks:** Triton `/v2/health/ready` and model repository; ONNX
  checksum vs the active release.
- **Remediation:** rematerialize the model repository
  (`ranking-triton-materializer`) or roll back the release.

## RankingRuntimeUnauthenticatedConnections

- **Symptom:** the ranking coordinator or runner accepted connections without
  the HMAC handshake in the last 15 minutes
  (`video_commerce_ranking_runtime_auth_total{result="unauthenticated"}`).
- **Impact:** `RANKING_RUNTIME_AUTH_MODE=permissive` is letting unauthenticated
  peers use the ranking runtime protocol.
- **First checks:** whether `SECURITY_INTERNAL_SERVICE_KEY` is set on every
  ranking peer; any peer still running a release without the handshake.
- **Remediation:** set the key everywhere, confirm `result="authenticated"`
  only, then switch to `required`.

## RankingRuntimeAuthFailures

- **Symptom:** ranking peers fail the runtime handshake (`result` is
  `invalid_mac`, `expired`, `replayed`, `malformed`, `unsupported_version` or
  `rejected`).
- **Impact:** in `required` mode those connections are refused and ranking
  capacity drops.
- **First checks:** key mismatch during rotation, clock skew beyond ±60 s,
  or an unexpected client address.
- **Remediation:** roll the same key to every peer; fix NTP; block unknown
  clients with the ranking NetworkPolicy.

## RankingTritonOverloadSustained

- **Symptom:** the Triton adapter reports overloaded or in-flight rejections
  above one per second for 5 minutes.
- **Impact:** ranking requests are rejected under load.
- **First checks:** adapter in-flight, Triton queue time, dynamic batch sizes,
  CPU/GPU saturation.
- **Remediation:** add Triton capacity or tune batching; shed traffic.

## FlinkInteractionFeatureJobMissing

- **Symptom:** no running `video-commerce-interaction-features` job, or more
  than one.
- **Impact:** official online features stop updating (or are written twice).
  This inhibits `FlinkCheckpointUnavailable` and `FlinkInteractionWatermarkLag`.
- **First checks:** `scripts/flink_realtime_ops.sh verify`; JobManager logs.
- **Remediation:** restore exactly one job from the newest checkpoint
  ([Flink recovery](flink-realtime-recovery.md#restart-rebuild-and-jobmanager-loss)).

## FlinkRedisSnapshotConflict

- **Symptom:** the Flink Redis sink saw a snapshot version conflict.
- **Impact:** another writer is touching official feature keys.
- **First checks:** concurrent official writers (a second job, the Python
  feature-worker in the wrong mode); state epoch and generation.
- **Remediation:** stop the extra writer; follow
  [Flink recovery](flink-realtime-recovery.md#restart-rebuild-and-jobmanager-loss).

## FlinkRedisSnapshotStaleWrites

- **Symptom:** most snapshot writes in 15 minutes are rejected as stale.
- **Impact:** features stop advancing; the job likely started from empty state
  under an epoch that already wrote Redis.
- **First checks:** the job's restore source; `flink:feature_state_owner`.
- **Remediation:** restore from the latest checkpoint, or deliberately bump
  `FLINK_FEATURE_STATE_EPOCH`
  ([Flink recovery](flink-realtime-recovery.md#restart-rebuild-and-jobmanager-loss)).

## FlinkRedisSnapshotWriteFailure

- **Symptom:** more than three snapshot write failures in 10 minutes.
- **Impact:** official features fall behind; the job may restart.
- **First checks:** Redis availability and memory; Lua errors in TaskManager
  logs; restart count.
- **Remediation:** restore Redis; confirm checkpoints resume.

## FlinkInteractionWatermarkLag

- **Symptom:** the job's input watermark is more than 15 minutes behind.
- **Impact:** windowed features are stale; late events pile up.
- **First checks:** Kafka partition activity and idleness, consumer lag,
  late-event volume, producer clocks.
- **Remediation:** fix the stalled partition or producer clock; see
  [Late interactions](flink-realtime-recovery.md#late-interactions).

## FlinkCheckpointUnavailable

- **Symptom:** no successful checkpoint for more than 5 minutes.
- **Impact:** a restart would replay a long backlog; savepoint upgrades are
  unsafe.
- **First checks:** checkpoint storage health and permissions; checkpoint
  failure reasons in the Flink UI; backpressure.
- **Remediation:** restore checkpoint storage; pause replay and cutover until
  checkpoints succeed.

## FlinkLatePurchaseSustained

- **Symptom:** purchase events keep arriving after their realtime window.
- **Impact:** purchase features undercount until reconciled offline.
- **First checks:** the `late-interaction-events` quarantine topic; producer
  clocks; Kafka lag.
- **Remediation:** reconcile offline; never replay the quarantine into realtime
  windows ([Late interactions](flink-realtime-recovery.md#late-interactions)).

## RecommendationOutboxBacklog

- **Symptom:** the served-impression outbox has rows older than 5 minutes or
  more than 10000 pending.
- **Impact:** attribution coverage falls behind.
- **First checks:** Kafka health (`KafkaProducerDisconnected`); dispatcher
  errors in recommendation logs; Postgres latency.
- **Remediation:** restore Kafka/Postgres; the dispatcher drains on its own.

## RecommendationOutboxTerminal

- **Symptom:** outbox rows were retired without publishing (`reason` is
  `message_too_large`, `max_attempts` or `non_retryable`).
- **Impact:** those impressions never reach Kafka; their attribution and
  training labels are lost.
- **First checks:** `reason`; for `message_too_large`, the feature context size
  limit (`RECOMMENDATION_FEATURE_CONTEXT_MAX_BYTES`) and producer
  `max_request_size`; for `max_attempts`, the Kafka error history.
- **Remediation:** fix the cause; retired rows are kept for audit but are not
  re-sent automatically.

## RecommendationImpressionAttributionDegraded

- **Symptom:** more than 5% of served slates got `impression_tracking`
  `timeout` or `unavailable` for 15 minutes.
- **Impact:** recommendations are served without an impression ID, so they are
  excluded from attribution and training.
- **First checks:** Postgres latency (`DatabaseP95LatencyHigh`); outbox
  in-flight cap (`RECOMMENDATION_IMPRESSION_OUTBOX_MAX_INFLIGHT`) and budget
  (`RECOMMENDATION_IMPRESSION_OUTBOX_TIMEOUT_MS`).
- **Remediation:** fix database latency; raise the budget only after measuring
  the p99 impact.

## CatalogOutboxTerminal

- **Symptom:** catalog outbox rows were retired without publishing.
- **Impact:** downstream consumers miss those catalog activation events.
- **First checks:** `reason`; publisher logs; Kafka message size limits.
- **Remediation:** fix the cause and republish the activation if consumers
  need it.

## ContentTaskOutboxBacklog

- **Symptom:** content task outbox rows older than 5 minutes or more than 1000
  pending.
- **Impact:** uploads are durable but workers are not receiving them.
- **First checks:** `content-task-publisher` logs and heartbeat; Kafka health.
- **Remediation:** restore the publisher or broker
  ([Event And Artifact Reliability](event-and-artifact-reliability.md)).

## ContentUploadObjectMissing

- **Symptom:** a durable content job referenced a missing object and was
  terminalized as `failed_missing_object`.
- **Impact:** that upload will not be processed.
- **First checks:** object storage bucket, prefix and lifecycle rules; the
  orphan reconciliation grace period.
- **Remediation:** fix storage retention; ask the user to re-upload.

## ContentProcessingLeaseLost

- **Symptom:** a content worker lost its processing lease.
- **Impact:** the run is retried by another worker; repeated loss means work
  never finishes.
- **First checks:** worker stalls (CPU-bound inference), Postgres latency,
  processing duration vs lease length.
- **Remediation:** fix the stall; lengthen the lease only if processing is
  legitimately slower.

## ContentProcessingRedrivesExhausted

- **Symptom:** the reaper failed a run after `CONTENT_PROCESSING_MAX_REDRIVES`
  expired leases (`outcome="exhausted"`).
- **Impact:** that upload stays failed until an operator acts.
- **First checks:** worker logs for the content ID (OOM, crash, timeout);
  whether the same video kills workers repeatedly.
- **Remediation:** fix the worker or quarantine the input, then resubmit the
  upload.

## ContentZeroNormEmbeddings

- **Symptom:** CLIP produced (near) zero-norm visual embeddings.
- **Impact:** affected content is excluded from visual retrieval.
- **First checks:** content worker logs for CLIP errors; model files and
  device; input frame extraction.
- **Remediation:** fix the model or input path and reprocess the affected
  content.

## FeatureLakeMaterializationLagHigh

- **Symptom:** raw history materialization lag is above 2 minutes.
- **Impact:** PIT exports wait; shadow gates fail.
- **First checks:** the materializer Flink job, Kafka lag, Iceberg commit
  failures.
- **Remediation:** restore the materializer
  ([Offline Feature Store](offline-feature-store.md)).

## FeatureLakeWindowLagHigh

- **Symptom:** window snapshot materialization lag is above 15 minutes.
- **Impact:** window history is stale for PIT.
- **First checks:** as for
  [FeatureLakeMaterializationLagHigh](#featurelakematerializationlaghigh).
- **Remediation:** restore the window materialization path.

## FeatureLakeDlqTraffic

- **Symptom:** feature-lake events are rejected to the DLQ.
- **Impact:** history has gaps; PIT cutover must wait.
- **First checks:** contract validation failures in the DLQ records.
- **Remediation:** fix the producer contract, then replay.

## CatalogOutboxBacklog

- **Symptom:** the catalog outbox oldest row is older than 5 minutes or more
  than 10000 are pending.
- **Impact:** catalog activations do not reach Kafka.
- **First checks:** `catalog-event-publisher` logs and Kafka health.
- **Remediation:** restore the publisher or broker.

## FeatureLakeIcebergCommitFailure

- **Symptom:** an Iceberg atomic commit failed.
- **Impact:** materialization stalls; cutover must pause.
- **First checks:** Flink checkpoint and commit errors; catalog (iceberg-rest)
  and object storage health.
- **Remediation:** restore the catalog or storage and let the job retry from
  its checkpoint.

## PitManifestValidationFailure

- **Symptom:** a PIT manifest or Parquet shard failed validation.
- **Impact:** the trainer fails closed and does not fall back to legacy data.
- **First checks:** `reason` label; manifest and shard hashes
  ([PIT export and manifest](offline-feature-store.md#pit-export-and-manifest)).
- **Remediation:** re-export the run; never hand-edit a manifest.

## PitOnlineOfflineParityLow

- **Symptom:** online/offline feature parity is below 99.9% for 15 minutes.
- **Impact:** training would learn from features serving never sees.
- **First checks:** recent feature-store or Flink job changes; namespace
  mismatches.
- **Remediation:** keep the training source on legacy until parity recovers.

## PitLeakageDetected

- **Symptom:** PIT rows used future or unavailable features.
- **Impact:** offline metrics are inflated; the dataset is unsafe to train on.
- **First checks:** event-time vs availability columns in the export; clock
  clamping at ingest.
- **Remediation:** block training on the run; fix the join and re-export.

## PitDailyRunMissing

- **Symptom:** no successful PIT dataset run for more than 26 hours.
- **Impact:** training data goes stale.
- **First checks:** durable run state (`pit_materialization_runs`), Flink
  submission, manifest publication; `PitRunWaitingForMaterializer`.
- **Remediation:** fix the failing stage; the next run resumes the parked one.

## PitTrainerManifestFailure

- **Symptom:** training keeps failing although a manifest exists.
- **Impact:** no new ranking artifact.
- **First checks:** trainer logs for manifest, shard hash, schema or version
  errors.
- **Remediation:** fix the schema or version mismatch; re-export if the shard is
  bad ([Typed PIT shadow training](offline-feature-store.md#typed-pit-shadow-training)).

## PitOrchestratorRunStuck

- **Symptom:** a PIT run has held the orchestration lease for 2 hours.
- **Impact:** no other PIT run can start.
- **First checks:** the run's phase and Flink job; worker liveness.
- **Remediation:** let the lease expire and take over, or investigate the stuck
  phase.

## PitRunWaitingForMaterializer

- **Symptom:** the oldest run parked in `waiting_for_materializer` has a cutoff
  more than 6 hours old
  (`pit_orchestrator_waiting_for_materializer_age_seconds`).
- **Impact:** no new PIT dataset can be exported; `PitDailyRunMissing` follows.
- **First checks:** feature-history materializer job state and lag
  (`FeatureLakeMaterializationLagHigh`); the run's `last_error` (it names the
  lagging partition).
- **Remediation:** restore the materializer; the next orchestrator run resumes
  the parked run automatically.

## KnownUserFeaturesMissing

- **Symptom:** more than 5% of uncached recommendation requests read no `uf:`
  feature key for a known user, for 15 minutes.
- **Impact:** personalization degrades to cold-start behaviour.
- **First checks:** `FlinkInteractionFeatureJobMissing` and checkpoints;
  `FLINK_FEATURE_OUTPUT_NAMESPACE` vs the namespace serving reads; Redis
  evictions (`RedisMemoryHigh`).
- **Remediation:** restore the Flink job or namespace; rebuild official keys
  only via the documented epoch procedure
  ([Flink recovery](flink-realtime-recovery.md)).

## PitOrchestratorLeaseExpired

- **Symptom:** a running PIT job stopped renewing its lease.
- **Impact:** the run needs takeover.
- **First checks:** the orchestrator pod or container; its last logs.
- **Remediation:** start the orchestrator; it takes over expired leases.

## RankingArtifactFallbackPersisting

- **Symptom:** training produced an artifact in the last day, but serving still
  uses the untrained fallback.
- **Impact:** ranking quality is below the trained model.
- **First checks:** artifact activation failures, checksum sync, runner reload
  state ([Model artifact rolling activation](model-artifact-rolling-activation.md)).
- **Remediation:** fix activation and confirm convergence with
  `model_release.py wait-rollout`.

## ModelReleaseRegisteredBacklog

- **Symptom:** ranking releases have waited more than 24 hours for evaluation.
- **Impact:** new models are not promoted.
- **First checks:** PIT evidence, evaluator failures, slice sample thresholds.
- **Remediation:** see [Evaluate and promote](model-release-runbook.md#evaluate-and-promote).

## ModelReleaseStagingTooOld

- **Symptom:** a validated ranking release stays in staging for a day.
- **Impact:** a passed release is neither promoted nor retired.
- **First checks:** the gate evidence for the release.
- **Remediation:** promote or retire it
  ([Evaluate and promote](model-release-runbook.md#evaluate-and-promote)).

## ModelReleaseRunnerGenerationMismatch

- **Symptom:** a ranking runner has not converged to the active generation for
  10 minutes.
- **Impact:** mixed model generations serve traffic.
- **First checks:** artifact download, checksum validation and reload failures;
  `model_release.py rollout-status`.
- **Remediation:** fix the runner or roll back
  ([Model artifact rolling activation](model-artifact-rolling-activation.md)).

## RetrievalReleaseRegisteredBacklog

- **Symptom:** Two-Tower releases have waited more than 24 hours for
  evaluation.
- **Impact:** retrieval improvements are not promoted.
- **First checks:** retrieval PIT evidence, catalog coverage, slice thresholds.
- **Remediation:** see
  [Retrieval release runbook](retrieval-pit-release-runbook.md#train-and-evaluate).

## RetrievalReleaseStagingTooOld

- **Symptom:** a validated Two-Tower release stays in staging for a day.
- **Impact:** a passed release is neither promoted nor retired.
- **First checks:** the retrieval gate evidence.
- **Remediation:** promote or retire it
  ([Promote, observe, and roll back](retrieval-pit-release-runbook.md#promote-observe-and-roll-back)).

## ModelArtifactActivationFailures

- **Symptom:** a model artifact generation failed to activate.
- **Impact:** the previous verified generation keeps serving.
- **First checks:** `reason` label (checksum, schema, policy, smoke); service
  logs.
- **Remediation:** fix or retire the release
  ([Model artifact rolling activation](model-artifact-rolling-activation.md)).

## ModelArtifactGenerationCleanupFailure

- **Symptom:** local generation cleanup failed in the last hour.
- **Impact:** disk fills over time.
- **First checks:** filesystem capacity and protected generation references.
- **Remediation:** free space; do not delete generation directories while a
  service runs.

## IcebergMaintenanceFailure

- **Symptom:** an Iceberg maintenance run failed.
- **Impact:** snapshots and orphans accumulate; a later execute may be unsafe.
- **First checks:** the persisted plan and the failure in
  `iceberg_maintenance_runs`.
- **Remediation:** reconcile plan hash, table head and protected lineage before
  the next execute window ([Iceberg maintenance](iceberg-maintenance.md)).
