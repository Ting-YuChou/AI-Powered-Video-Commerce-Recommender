#!/usr/bin/env bash
set -euo pipefail

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-vc-flink-recovery-${RANDOM}}"
export FLINK_WATERMARK_IDLENESS_SECONDS="${FLINK_WATERMARK_IDLENESS_SECONDS:-5}"
ARTIFACT_DIR="${FLINK_RECOVERY_ARTIFACT_DIR:-/tmp/flink-recovery-artifacts}"
mkdir -p "$ARTIFACT_DIR"

cleanup() {
  docker compose --profile flink logs --no-color >"$ARTIFACT_DIR/compose.log" 2>&1 || true
  curl -fsS http://127.0.0.1:8081/jobs/overview >"$ARTIFACT_DIR/jobs.json" 2>/dev/null || true
  docker compose --profile flink down -v --remove-orphans >/dev/null 2>&1 || true
}
report_failure() {
  local line="$1" status="$2"
  echo "Flink recovery drill failed at line ${line} (exit ${status})" >&2
}
trap 'report_failure "$LINENO" "$?"' ERR
trap cleanup EXIT

docker compose --profile flink up -d --build \
  redis postgres zookeeper kafka kafka-init flink-jobmanager flink-taskmanager flink-interaction-features

docker compose exec -T postgres psql -v ON_ERROR_STOP=1 \
  -U video_commerce -d video_commerce >/dev/null <<'SQL'
CREATE TABLE interaction_events (
  event_id VARCHAR(64) PRIMARY KEY,
  schema_version INTEGER NOT NULL DEFAULT 1,
  request_id VARCHAR(64),
  user_id VARCHAR(255) NOT NULL,
  product_id VARCHAR(255) NOT NULL,
  action VARCHAR(64) NOT NULL,
  context JSON NOT NULL DEFAULT '{}'::json,
  occurred_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
SQL

for _ in $(seq 1 180); do
  if scripts/flink_realtime_ops.sh verify >/dev/null 2>&1; then break; fi
  sleep 2
done
scripts/flink_realtime_ops.sh verify

now="$(python3 -c 'import time; print(time.time())')"
produce() {
  local event_id="$1" action="$2" occurred="$3"
  printf 'recovery-user:{"event_id":"%s","schema_version":1,"user_id":"recovery-user","product_id":"p1","action":"%s","occurred_at":%s,"context":{}}\n' "$event_id" "$action" "$occurred" |
    docker compose exec -T kafka kafka-console-producer \
      --bootstrap-server kafka:9092 --topic user-interactions \
      --property parse.key=true --property key.separator=:
}

produce recovery-1 click "$now"
for _ in $(seq 1 60); do
  generation="$(docker compose exec -T redis redis-cli HGET ufm:recovery-user generation | tr -d '\r')"
  [[ "$generation" == "1" ]] && break
  sleep 2
done
[[ "${generation:-}" == "1" ]]

savepoint="$(scripts/flink_realtime_ops.sh stop-with-savepoint)"
[[ -n "$savepoint" ]]
scripts/flink_realtime_ops.sh restore "$savepoint"
for _ in $(seq 1 60); do
  if scripts/flink_realtime_ops.sh verify >/dev/null 2>&1; then break; fi
  sleep 2
done
scripts/flink_realtime_ops.sh verify

produce recovery-1 click "$now"
future="$(python3 -c "print(float('$now') + 7200)")"
late="$(python3 -c "print(float('$now') - 7200)")"
produce recovery-advance view "$future"
# Allow idle partitions to stop holding back the watermark before the late purchase.
sleep 10
produce recovery-late purchase "$late"

for _ in $(seq 1 60); do
  generation="$(docker compose exec -T redis redis-cli HGET ufm:recovery-user generation | tr -d '\r')"
  [[ "$generation" == "3" ]] && break
  sleep 2
done
[[ "${generation:-}" == "3" ]]

for _ in $(seq 1 60); do
  row_count="$(docker compose exec -T postgres psql -U video_commerce -d video_commerce -Atc "SELECT count(*) FROM interaction_events WHERE user_id='recovery-user'" | tr -d '\r')"
  [[ "$row_count" == "3" ]] && break
  sleep 2
done
[[ "$row_count" == "3" ]]

docker compose exec -T kafka timeout 30 kafka-console-consumer \
  --bootstrap-server kafka:9092 --topic late-interaction-events --from-beginning --max-messages 1 \
  >"$ARTIFACT_DIR/late-event.json"
python3 -c 'import json,sys; event=json.load(open(sys.argv[1])); assert event["event_type"]=="late_interaction"; assert event["source_event_id"]=="recovery-late"' "$ARTIFACT_DIR/late-event.json"

dlq_time="$(python3 -c 'import time; print(time.time())')"
printf '{"schema_version":2,"event_id":"drill-dlq","source_topic":"user-interactions","source_partition":0,"source_offset":99,"source_key":"recovery-user","source_value":{"event_id":"recovery-dlq","schema_version":1,"user_id":"recovery-user","product_id":"p2","action":"click","occurred_at":%s,"context":{}},"source_headers":[],"occurred_at":%s}\n' "$dlq_time" "$dlq_time" |
  docker compose exec -T kafka kafka-console-producer --bootstrap-server kafka:9092 --topic dead-letter-events

docker run --rm \
  --network "${COMPOSE_PROJECT_NAME}_video-commerce-network" \
  -v "$PWD:/app:ro" -v "$ARTIFACT_DIR:/artifacts" -w /app python:3.10-slim \
  sh -ec '
    pip install -q aiokafka
    python scripts/replay_kafka_dlq.py --bootstrap-servers kafka:9092 --allowed-source-topic user-interactions --audit-output /artifacts/dlq-dry-run.json
    python scripts/replay_kafka_dlq.py --bootstrap-servers kafka:9092 --allowed-source-topic user-interactions --execute --audit-output /artifacts/dlq-execute-1.json
    python scripts/replay_kafka_dlq.py --bootstrap-servers kafka:9092 --allowed-source-topic user-interactions --execute --audit-output /artifacts/dlq-execute-2.json
  '

for _ in $(seq 1 60); do
  generation="$(docker compose exec -T redis redis-cli HGET ufm:recovery-user generation | tr -d '\r')"
  [[ "$generation" == "4" ]] && break
  sleep 2
done
[[ "${generation:-}" == "4" ]]
scripts/flink_realtime_ops.sh verify

# Routine restart without a savepoint (./startup.sh restart is compose down/up):
# the submitter must restore the newest retained checkpoint so generations stay
# continuous instead of restarting at 1 and crash-looping on digest conflicts.
job_id="$(scripts/flink_realtime_ops.sh verify | awk '{print $2}')"
for _ in $(seq 1 60); do
  completed="$(curl -fsS "http://127.0.0.1:8081/jobs/${job_id}/checkpoints" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("latest") or {}).get("completed") is not None)')"
  [[ "$completed" == "True" ]] && break
  sleep 2
done
[[ "$completed" == "True" ]]
# Let a checkpoint that includes generation 4 complete.
sleep 35
docker compose --profile flink down --remove-orphans
docker compose --profile flink up -d \
  redis postgres zookeeper kafka kafka-init flink-jobmanager flink-taskmanager flink-interaction-features
for _ in $(seq 1 180); do
  if scripts/flink_realtime_ops.sh verify >/dev/null 2>&1; then break; fi
  sleep 2
done
scripts/flink_realtime_ops.sh verify
# Capture first: grep -q exits early and SIGPIPEs `docker compose logs` under pipefail.
docker compose logs --no-color flink-interaction-features >"$ARTIFACT_DIR/submitter-after-restart.log" 2>&1
grep -q "restoring video-commerce-interaction-features from latest retained checkpoint" \
  "$ARTIFACT_DIR/submitter-after-restart.log"

produce recovery-after-restart click "$(python3 -c 'import time; print(time.time())')"
for _ in $(seq 1 60); do
  generation="$(docker compose exec -T redis redis-cli HGET ufm:recovery-user generation | tr -d '\r')"
  [[ "$generation" == "5" ]] && break
  sleep 2
done
[[ "${generation:-}" == "5" ]]
# Still RUNNING (not RESTARTING) after the post-restart write.
sleep 10
scripts/flink_realtime_ops.sh verify
echo "Flink recovery drill passed"
