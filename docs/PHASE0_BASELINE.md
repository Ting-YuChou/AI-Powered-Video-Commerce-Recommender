# Phase 0 Baseline and SLO Definition

## Scope

This baseline focuses on three API paths:

1. `POST /api/interactions`
2. `POST /api/recommendations`
3. Mixed traffic: `90% /api/interactions + 10% /api/recommendations`

`POST /api/content/upload` is excluded from the 10,000 RPS target because it is a background-processing workflow, not a latency-sensitive serving path.

## Baseline Traffic Assumptions

### Recommendation Requests

- Endpoint: `POST /api/recommendations`
- Payload shape:

```json
{
  "user_id": "user_000123",
  "content_id": "content_000045",
  "context": {
    "device": "mobile",
    "page": "home",
    "session_position": 3,
    "time_on_page": 42
  },
  "k": 20
}
```

- User cardinality: `100,000`
- Product cardinality: `1,000,000`
- `content_id` hit ratio:
  - `70%` with `content_id`
  - `30%` without `content_id`
- Recommendation cache assumption for baseline:
  - Cold run: `0%`
  - Warm run: `60%`
- User feature cache assumption:
  - Warm Redis keys exist for `95%` of active users

### Interaction Requests

- Endpoint: `POST /api/interactions`
- Payload shape:

```json
{
  "user_id": "user_000123",
  "product_id": "prod_001245",
  "action": "click",
  "context": {
    "page": "video_recommendations",
    "recommendation_position": 2,
    "session_id": "sess_123"
  }
}
```

- User cardinality: `100,000`
- Product cardinality: `1,000,000`
- Action distribution:
  - `view`: `70%`
  - `click`: `20%`
  - `add_to_cart`: `7%`
  - `purchase`: `3%`
- Kafka assumption:
  - Producer enabled
  - Consumer lag must be measured on worker consumer groups during mixed tests

## Baseline SLOs

### Interaction Path

- Target throughput: `10,000 RPS`
- Error rate: `< 0.5%`
- Latency:
  - `P50 < 20ms`
  - `P95 < 50ms`
  - `P99 < 100ms`

### Recommendation Path

- Initial target throughput: `1,000-2,000 RPS`
- Error rate: `< 1%`
- Latency:
  - `P50 < 80ms`
  - `P95 < 150ms`
  - `P99 < 300ms`

## Metrics to Collect

Collect all of the following in every run:

- RPS
- `P50`, `P95`, `P99`
- Error rate
- API process CPU
- API process memory
- Redis `instantaneous_ops_per_sec`
- Redis connected clients
- Kafka producer connectivity
- Kafka consumer lag on worker consumer groups

## Endpoints Added for Phase 0

- Prometheus metrics: `GET /metrics`
- JSON system summary: `GET /metrics/system`
- Health: `GET /health`

The application now also emits:

- `X-Request-ID` response header
- Structured JSON logs for request completion and failures

## Suggested Baseline Procedure

1. Start the stack.
2. Confirm `GET /health` is healthy.
3. Confirm `GET /metrics` returns Prometheus text format.
4. Run cold-cache tests.
5. Run warm-cache tests.
6. Capture Prometheus samples, k6 summary, and system logs together.

## k6 Commands

### Recommendation Baseline

```bash
k6 run loadtest/k6/recommendations.js \
  -e BASE_URL=http://localhost:8000 \
  -e VUS=50 \
  -e DURATION=60s \
  -e USER_POOL=100000 \
  -e CONTENT_POOL=10000 \
  -e CONTENT_HIT_RATIO=0.7
```

### Interaction Baseline

```bash
k6 run loadtest/k6/interactions.js \
  -e BASE_URL=http://localhost:8000 \
  -e VUS=200 \
  -e DURATION=60s \
  -e USER_POOL=100000 \
  -e PRODUCT_POOL=1000000
```

### Mixed Baseline

```bash
k6 run loadtest/k6/mixed.js \
  -e BASE_URL=http://localhost:8000 \
  -e VUS=200 \
  -e DURATION=60s \
  -e USER_POOL=100000 \
  -e PRODUCT_POOL=1000000 \
  -e CONTENT_POOL=10000
```

## Phase 0 Exit Criteria

Phase 0 is complete when:

1. The baseline traffic profile is fixed and documented.
2. Every request carries a request ID.
3. Logs are structured and machine-parsable.
4. Prometheus can scrape latency and throughput metrics.
5. One complete cold run and one complete warm run have been captured for:
   - interactions
   - recommendations
   - mixed traffic
