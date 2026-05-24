#!/usr/bin/env bash
set -euo pipefail

# Start the official Flink feature pipeline and enable ranking consumption of
# Flink realtime window features. Shadow promotion remains an explicit
# compatibility option, not part of the normal production path.

export FEATURE_PIPELINE_MODE="${FEATURE_PIPELINE_MODE:-flink}"
export FLINK_FEATURE_OUTPUT_NAMESPACE="${FLINK_FEATURE_OUTPUT_NAMESPACE:-official}"
export RANKING_REALTIME_WINDOW_FEATURES_ENABLED="${RANKING_REALTIME_WINDOW_FEATURES_ENABLED:-true}"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  docker compose --profile flink config -q
  echo "FEATURE_PIPELINE_MODE=${FEATURE_PIPELINE_MODE}"
  echo "FLINK_FEATURE_OUTPUT_NAMESPACE=${FLINK_FEATURE_OUTPUT_NAMESPACE}"
  echo "RANKING_REALTIME_WINDOW_FEATURES_ENABLED=${RANKING_REALTIME_WINDOW_FEATURES_ENABLED}"
  exit 0
fi

if [[ "${PROMOTE_FLINK_SHADOW:-false}" == "true" ]]; then
  python scripts/promote_flink_feature_shadow.py --execute
fi

docker compose --profile flink up -d --build \
  flink-jobmanager \
  flink-taskmanager \
  flink-interaction-features \
  recommendation-service \
  ranking-service \
  gateway-api

docker compose ps flink-jobmanager flink-taskmanager flink-interaction-features recommendation-service ranking-service gateway-api
