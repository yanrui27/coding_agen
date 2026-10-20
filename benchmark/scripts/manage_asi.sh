#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/asi_env.sh

case "${1:-}" in
  docker)
    shift
    asi_start_docker
    exec docker "$@"
    ;;
  prepare-images)
    shift
    asi_start_docker
    exec .venv-harbor/bin/python -B -u scripts/prepare_asi_images.py "$@"
    ;;
  cli)
    shift
    exec .venv-harbor/bin/python -B -u scripts/eval_asi.py "$@"
    ;;
  *)
    echo 'Usage: bash scripts/manage_asi.sh {docker ARGS...|prepare-images|cli ARGS...}' >&2
    exit 2
    ;;
esac
