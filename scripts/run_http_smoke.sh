#!/usr/bin/env bash
# Fresh-stack HTTP smoke: edge -> gateway -> recommendation/ingest, plus an
# outage check that the edge stays up while the ranking coordinator is down.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

project_name="${COMPOSE_PROJECT_NAME:-vc-http-smoke-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"
artifact_dir="${HTTP_SMOKE_ARTIFACT_DIR:-/tmp/http-smoke-artifacts}"
compose=(docker compose -p "$project_name")

export COMPOSE_PROJECT_NAME="$project_name"
export ENVIRONMENT=test
export SECURITY_AUTH_MODE=disabled
export VECTOR_BOOTSTRAP_MODE=sample
export SERVICE_GATEWAY_WORKERS=1
export SERVICE_RECOMMENDATION_WORKERS=1
export SERVICE_RECOMMENDATION_REPLICAS=1
export SERVICE_RECOMMENDATION_SERVICE_URLS=http://recommendation-service:8001
export SERVICE_INTERACTION_WORKERS=1
export SERVICE_RANKING_WORKERS=1
export RANKING_SERVICE_REPLICAS=1
# One runner until SystemStore schema creation is serialized (Stage B WS-2).
export RANKING_RUNNER_REPLICAS=1
export SERVICE_RANKING_MIN_HEALTHY_RUNNERS=1
export RANKING_BATCH_RUNNER_COUNT=1
export RANKING_INFERENCE_EXECUTOR_WORKERS=1
export RANKING_ALLOW_UNTRAINED_FALLBACK=true
export RANKING_REQUIRE_VERIFIED_ARTIFACT=false
export RECOMMENDATION_KNOWN_USER_SNAPSHOT_ENABLED=false
export MONITORING_ENABLE_TRACING=false

collect_diagnostics() {
  mkdir -p "$artifact_dir"
  "${compose[@]}" ps --all >"$artifact_dir/compose-ps.txt" 2>&1 || true
  "${compose[@]}" logs --no-color >"$artifact_dir/compose.log" 2>&1 || true
  "${compose[@]}" exec -T gateway-api curl -sS http://127.0.0.1:8000/readyz/full \
    >"$artifact_dir/gateway-readyz-full.json" 2>&1 || true
}

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    collect_diagnostics
  fi
  "${compose[@]}" --profile demo --profile test down -v --remove-orphans \
    >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

wait_for_gateway_ready() {
  local deadline=$((SECONDS + ${1:-240}))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "${compose[@]}" exec -T gateway-api \
      curl -fsS http://127.0.0.1:8000/readyz >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
  echo "gateway /readyz did not return 200" >&2
  return 1
}

run_integration() {
  "${compose[@]}" --profile test run --rm --no-deps \
    -e EDGE_BASE_URL=http://caddy "$@" \
    integration-tests pytest -q tests/integration/test_http_smoke.py
}

"${compose[@]}" --profile demo build vector-sample-bootstrap
"${compose[@]}" --profile demo run --rm vector-sample-bootstrap
"${compose[@]}" up -d --build --wait --wait-timeout 180 interaction-ingest-service
"${compose[@]}" up -d --build --wait --wait-timeout 240 ranking-coordinator
# Recommendation stays degraded with an untrained ranker, so do not --wait on it.
"${compose[@]}" up -d --build gateway-api caddy
wait_for_gateway_ready 300
"${compose[@]}" --profile test build integration-tests

run_integration

# A shared dependency outage must not take the edge or ingest down.
"${compose[@]}" stop ranking-coordinator
run_integration -e INTEGRATION_EXPECT_RANKING_DOWN=1
"${compose[@]}" start ranking-coordinator
