#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${FLINK_INTERACTION_JOB_NAME:-video-commerce-interaction-features}"
BASE_URL="${FLINK_REST_URL:-http://127.0.0.1:8081}"
SAVEPOINT_DIR="${FLINK_SAVEPOINT_DIR:-file:///flink/savepoints/runtime}"

active_jobs() {
  curl -fsS "${BASE_URL}/jobs/overview" | python3 -c '
import json,sys
name=sys.argv[1]
for job in json.load(sys.stdin).get("jobs",[]):
    if job.get("name")==name and job.get("state") in {"RUNNING","RESTARTING"}:
        print(job["jid"])
' "$JOB_NAME"
}

one_job() {
  local jobs
  jobs="$(active_jobs)"
  local count
  count="$(grep -c . <<<"$jobs" || true)"
  if [[ "$count" -ne 1 ]]; then
    echo "expected exactly one active ${JOB_NAME} job, found ${count}" >&2
    return 1
  fi
  printf '%s\n' "$jobs"
}

poll_operation() {
  local url="$1"
  for _ in $(seq 1 120); do
    local response status
    response="$(curl -fsS "$url")"
    status="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",{}).get("id",""))' <<<"$response")"
    if [[ "$status" == "COMPLETED" ]]; then
      python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("operation",{}).get("location") or r.get("operation",{}).get("savepoint",{}).get("location") or "")' <<<"$response"
      return 0
    fi
    [[ "$status" != "FAILED" ]] || { printf '%s\n' "$response" >&2; return 1; }
    sleep 2
  done
  echo "Flink operation timed out: $url" >&2
  return 1
}

command="${1:-status}"
case "$command" in
  status) job_id="$(one_job)"; curl -fsS "${BASE_URL}/jobs/${job_id}" ;;
  verify)
    job_id="$(one_job)"
    state="$(curl -fsS "${BASE_URL}/jobs/${job_id}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state"))')"
    [[ "$state" == "RUNNING" || "$state" == "RESTARTING" ]]
    echo "${JOB_NAME} ${job_id} ${state}"
    ;;
  savepoint|stop-with-savepoint)
    job_id="$(one_job)"
    if [[ "$command" == "savepoint" ]]; then endpoint="savepoints"; body="{\"target-directory\":\"${SAVEPOINT_DIR}\",\"cancel-job\":false}"
    else endpoint="stop"; body="{\"targetDirectory\":\"${SAVEPOINT_DIR}\",\"drain\":false}"; fi
    request_id="$(curl -fsS -X POST -H 'Content-Type: application/json' -d "$body" "${BASE_URL}/jobs/${job_id}/${endpoint}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["request-id"])')"
    # Flink reports both trigger-savepoint and stop-with-savepoint operations
    # through the savepoints status endpoint.
    poll_operation "${BASE_URL}/jobs/${job_id}/savepoints/${request_id}"
    ;;
  restore)
    savepoint="${2:-}"
    [[ -n "$savepoint" ]] || { echo "usage: $0 restore <savepoint-path>" >&2; exit 2; }
    [[ -z "$(active_jobs)" ]] || { echo "refusing restore while ${JOB_NAME} is active" >&2; exit 1; }
    FLINK_RESTORE_SAVEPOINT_PATH="$savepoint" docker compose --profile flink run --rm --no-deps flink-interaction-features
    one_job >/dev/null
    ;;
  *) echo "usage: $0 {status|savepoint|stop-with-savepoint|restore PATH|verify}" >&2; exit 2 ;;
esac
