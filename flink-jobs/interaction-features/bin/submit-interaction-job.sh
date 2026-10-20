#!/bin/sh
# Submit the interaction feature job once, restoring state whenever possible.
#
# Restore order:
#   1. FLINK_RESTORE_SAVEPOINT_PATH (explicit savepoint/checkpoint path)
#   2. the newest retained checkpoint under FLINK_CHECKPOINT_DIR (file:// only),
#      unless FLINK_RESTORE_FROM_LATEST_CHECKPOINT=false
#   3. empty state
#
# Starting from empty state under an epoch that already wrote Redis is refused by
# the job itself (UserFeatureRedisSink epoch owner check). To discard state on
# purpose, bump FLINK_FEATURE_STATE_EPOCH and set
# FLINK_RESTORE_FROM_LATEST_CHECKPOINT=false. See
# docs/operations/flink-realtime-recovery.md.
set -eu

JOBMANAGER="${FLINK_JOBMANAGER_ADDRESS:-flink-jobmanager:8081}"
JOB_NAME="${FLINK_INTERACTION_JOB_NAME:-video-commerce-interaction-features}"
JAR="${FLINK_INTERACTION_JOB_JAR:-/opt/flink/usrlib/interaction-features.jar}"
CHECKPOINT_DIR="${FLINK_CHECKPOINT_DIR:-file:///flink/checkpoints/runtime}"

running_count() {
  flink list -m "$JOBMANAGER" -r 2>/dev/null | grep -c "$JOB_NAME" || true
}

latest_checkpoint() {
  case "$CHECKPOINT_DIR" in
    file://*) root="${CHECKPOINT_DIR#file://}" ;;
    /*) root="$CHECKPOINT_DIR" ;;
    *) return 0 ;;  # remote storage: pass FLINK_RESTORE_SAVEPOINT_PATH explicitly
  esac
  [ -d "$root" ] || return 0
  # Only completed checkpoints have a _metadata file; newest by mtime wins.
  metadata="$(ls -1t "$root"/*/chk-*/_metadata 2>/dev/null | head -n 1 || true)"
  [ -n "$metadata" ] && dirname "$metadata"
  return 0
}

until flink list -m "$JOBMANAGER" >/dev/null 2>&1; do sleep 2; done

count="$(running_count)"
if [ "$count" -gt 1 ]; then
  echo "multiple running ${JOB_NAME} jobs detected" >&2
  exit 1
fi

if [ "$count" -eq 0 ]; then
  restore=""
  if [ -n "${FLINK_RESTORE_SAVEPOINT_PATH:-}" ]; then
    restore="$FLINK_RESTORE_SAVEPOINT_PATH"
    echo "restoring ${JOB_NAME} from explicit path ${restore}"
  elif [ "${FLINK_RESTORE_FROM_LATEST_CHECKPOINT:-true}" = "true" ]; then
    restore="$(latest_checkpoint)"
    if [ -n "$restore" ]; then
      echo "restoring ${JOB_NAME} from latest retained checkpoint ${restore}"
    fi
  fi
  if [ -z "$restore" ]; then
    echo "starting ${JOB_NAME} from empty state (epoch ${FLINK_FEATURE_STATE_EPOCH:-1})"
  fi
  flink run -d -m "$JOBMANAGER" ${restore:+-s "$restore"} \
    -c com.videocommerce.flink.InteractionFeatureJob "$JAR"
fi

until [ "$(running_count)" -eq 1 ]; do sleep 2; done
