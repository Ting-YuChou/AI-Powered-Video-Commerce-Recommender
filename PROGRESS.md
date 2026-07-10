# Progress

## 2026-07-09 — Ranking PIT feature-store foundation

- Added a shared versioned ranking feature contract and deterministic `as_of_ts`
  extraction for training; PIT rows now override current online user/item maps.
- Governed public interaction timestamps with `event_time`, immutable
  `server_received_at`, 30-day historical and 5-minute future bounds, while
  retaining `timestamp` as a backwards-compatible alias. Flink prefers the new
  event time.
- Added feature-lake rollout configuration, a manifest-validated PIT dataset
  reader, and a trainer path that fails closed in `FEATURE_LAKE_TRAINING_SOURCE=pit`
  rather than falling back to Redis/Postgres current-state features.
- Upgraded the Flink project/runtime configuration to 1.20.1, added the Iceberg
  1.11 runtime, and added a batch PIT SQL job with feature event/availability
  predicates and the 7-day attribution window. Helm now leaves the legacy
  Python feature worker disabled by default under the Flink-authoritative path.
- Key files: `video_commerce/common/event_time.py`,
  `video_commerce/ml/point_in_time.py`,
  `video_commerce/ml/pit_training_dataset.py`,
  `video_commerce/services/model_trainer/main.py`,
  `flink-jobs/interaction-features/`.
- Verification: Docker backend tests for event time, config, PIT contracts,
  trainer routing, ranking LTR/features; isolated Maven container `mvn test`;
  `docker compose config -q`.
- Follow-up: materialize Kafka interaction/catalog records into the declared
  Iceberg history tables and publish the completed snapshot as the manifest
  export consumed by the trainer; then run an end-to-end MinIO/REST catalog
  integration test before enabling `FEATURE_LAKE_TRAINING_SOURCE=pit`.
