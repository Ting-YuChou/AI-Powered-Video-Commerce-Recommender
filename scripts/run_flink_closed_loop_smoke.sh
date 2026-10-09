#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

project_name="${COMPOSE_PROJECT_NAME:-vc-flink-closed-loop-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"
artifact_dir="${FLINK_CLOSED_LOOP_ARTIFACT_DIR:-/tmp/flink-closed-loop-artifacts}"
compose=(docker compose -p "$project_name")

export COMPOSE_PROJECT_NAME="$project_name"
export ENVIRONMENT=test
export VECTOR_BOOTSTRAP_MODE=sample
export FEATURE_PIPELINE_MODE=flink
export FLINK_FEATURE_OUTPUT_NAMESPACE=official
export SERVICE_GATEWAY_WORKERS=1
export SERVICE_GATEWAY_REPLICAS=1
export SERVICE_RECOMMENDATION_WORKERS=1
export SERVICE_RECOMMENDATION_REPLICAS=1
export SERVICE_RECOMMENDATION_SERVICE_URLS=http://recommendation-service:8001
export SERVICE_INTERACTION_WORKERS=1
export SERVICE_RANKING_WORKERS=1
export RANKING_SERVICE_REPLICAS=1
export RANKING_RUNNER_REPLICAS=1
export SERVICE_RANKING_MIN_HEALTHY_RUNNERS=1
export RANKING_BATCH_RUNNER_COUNT=1
export RANKING_INFERENCE_EXECUTOR_WORKERS=1
export RANKING_ALLOW_UNTRAINED_FALLBACK=true
export RANKING_REQUIRE_VERIFIED_ARTIFACT=false
export RECOMMENDATION_IMPRESSION_LOGGING_ENABLED=true
export RECOMMENDATION_KNOWN_USER_SNAPSHOT_ENABLED=false
export MONITORING_ENABLE_TRACING=false

collect_diagnostics() {
  mkdir -p "$artifact_dir"
  "${compose[@]}" ps --all >"$artifact_dir/compose-ps.txt" 2>&1 || true
  "${compose[@]}" logs --no-color >"$artifact_dir/compose.log" 2>&1 || true
  curl -sS http://127.0.0.1:8081/jobs/overview \
    >"$artifact_dir/flink-jobs-overview.json" 2>"$artifact_dir/flink-rest-error.txt" || true
  "${compose[@]}" exec -T gateway-api \
    curl -sS http://127.0.0.1:8000/readyz \
    >"$artifact_dir/gateway-readiness.json" 2>"$artifact_dir/gateway-readiness-error.txt" || true
}

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    collect_diagnostics
  fi
  "${compose[@]}" --profile flink --profile demo --profile test down \
    -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

"${compose[@]}" --profile demo build vector-sample-bootstrap
"${compose[@]}" --profile demo run --rm vector-sample-bootstrap
"${compose[@]}" up -d --build --wait --wait-timeout 180 \
  interaction-ingest-service
"${compose[@]}" up -d --build --wait --wait-timeout 240 \
  ranking-coordinator
"${compose[@]}" --profile flink up -d --build \
  gateway-api flink-interaction-features

deadline=$((SECONDS + 240))
while [ "$SECONDS" -lt "$deadline" ]; do
  overview="$(curl -fsS http://127.0.0.1:8081/jobs/overview 2>/dev/null || true)"
  if [ -n "$overview" ] && JOB_OVERVIEW="$overview" python -c \
    'import json, os, sys; jobs=json.loads(os.environ["JOB_OVERVIEW"]).get("jobs", []); matches=[job for job in jobs if job.get("name") == "video-commerce-interaction-features" and job.get("state") in {"RUNNING", "RESTARTING"}]; sys.exit(0 if len(matches) == 1 else 1)'; then
    break
  fi
  sleep 2
done

overview="$(curl -fsS http://127.0.0.1:8081/jobs/overview)"
JOB_OVERVIEW="$overview" python -c \
  'import json, os; jobs=json.loads(os.environ["JOB_OVERVIEW"]).get("jobs", []); matches=[job for job in jobs if job.get("name") == "video-commerce-interaction-features" and job.get("state") in {"RUNNING", "RESTARTING"}]; assert len(matches) == 1, jobs'

"${compose[@]}" --profile test run --rm --build --no-deps \
  -e RUN_FLINK_CLOSED_LOOP_TESTS=1 \
  integration-tests \
  pytest -q tests/integration/test_flink_closed_loop.py
