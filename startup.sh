#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_HEALTH_URL="${APP_HEALTH_URL:-http://localhost/readyz}"

print() {
  local level="$1"
  shift
  printf '[%s] %s\n' "$level" "$*"
}

run_compose() {
  (
    cd "$PROJECT_DIR"
    docker compose "$@"
  )
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    print ERROR "Missing required command: $1"
    exit 1
  fi
}

show_usage() {
  cat <<'EOF'
Usage: ./startup.sh <command> [service]

Commands:
  start      Build and start the production compose stack
  stop       Stop the stack and remove containers
  restart    Rebuild and restart the stack
  status     Show compose service status
  logs       Follow logs for the whole stack or one service
  health     Query the public readiness endpoint
  build      Build images only
EOF
}

start_stack() {
  print INFO "Starting microservice stack with Docker Compose"
  run_compose up -d --build
  print INFO "Public edge is expected at http://localhost"
  print INFO "Run './startup.sh health' once containers are healthy"
}

stop_stack() {
  print INFO "Stopping microservice stack"
  run_compose down
}

restart_stack() {
  stop_stack
  start_stack
}

show_status() {
  run_compose ps
}

show_logs() {
  local service="${1:-}"
  if [[ -n "$service" ]]; then
    run_compose logs -f "$service"
  else
    run_compose logs -f
  fi
}

check_health() {
  require_command curl
  print INFO "Checking ${DEFAULT_HEALTH_URL}"
  curl --fail --silent --show-error "$DEFAULT_HEALTH_URL"
  printf '\n'
}

build_images() {
  print INFO "Building images"
  run_compose build
}

main() {
  require_command docker

  local command="${1:-start}"
  case "$command" in
    start)
      start_stack
      ;;
    stop)
      stop_stack
      ;;
    restart)
      restart_stack
      ;;
    status)
      show_status
      ;;
    logs)
      show_logs "${2:-}"
      ;;
    health)
      check_health
      ;;
    build)
      build_images
      ;;
    *)
      show_usage
      exit 1
      ;;
  esac
}

main "$@"
