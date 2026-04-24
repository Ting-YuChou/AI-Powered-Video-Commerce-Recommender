# Phase 1 Service Topology

This repository now includes a split-service Phase 1 topology instead of a single
serving process:

- `gateway-api`
  - Validates requests
  - Enforces optional API-key auth
  - Applies per-client in-memory rate limiting
  - Routes recommendation requests to `recommendation-service`
  - Routes interaction requests to `interaction-ingest-service`
  - Accepts content uploads and enqueues video processing tasks

- `recommendation-service`
  - Loads the serving-time ANN/vector index
  - Loads the ranking model
  - Reads Redis-backed user/content features
  - Reads/writes recommendation cache
  - Does not retrain models inside the serving process

- `interaction-ingest-service`
  - Accepts interaction events quickly
  - Enqueues to Kafka and waits for broker acknowledgment before returning `202`
  - Returns `503` when Kafka is unavailable so clients can retry without silent data loss
  - Does not synchronously update user features on the request path

- `content-worker`
  - Consumes video processing tasks from Kafka
  - Performs CLIP/OCR/video feature extraction offline
  - Updates Redis and vector search state

- `feature-worker`
  - Consumes interaction events from Kafka
  - Performs async feature updates outside the ingest request path

- `model-trainer`
  - Runs periodic recommendation model updates offline
  - Keeps retraining out of the live recommendation service

## Entry Points

- Gateway: [gateway_api.py](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/gateway_api.py)
- Recommendation service: [recommendation_api.py](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/recommendation_api.py)
- Interaction ingest service: [interaction_ingest_api.py](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/interaction_ingest_api.py)
- Model trainer: [model_trainer.py](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/model_trainer.py)
- Shared service runtime helpers: [service_common.py](/Users/zhoutingyou/Desktop/video_commerce/AI.Video.Commerce.Code/service_common.py)

## Deployment Notes

- The public port is `8000` on `gateway-api`.
- Internal service ports are `8001` for `recommendation-service` and `8002` for `interaction-ingest-service`.
- Worker communication uses Kafka.
- Redis remains a single node in local compose for simplicity; production should replace it with Redis Cluster / managed equivalent.
- Kafka consumers use manual offset commits by default; handler success or DLQ publish is required before offsets are committed.
- `docker-compose.yml` removes the old single `app` service and deploys the split topology.
