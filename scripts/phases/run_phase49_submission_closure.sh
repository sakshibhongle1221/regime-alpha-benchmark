#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="python"
if [ -x "./env/Scripts/python.exe" ]; then
  PYTHON="./env/Scripts/python.exe"
elif [ -x "./venv/bin/python" ]; then
  PYTHON="./venv/bin/python"
elif [ -x "./env/bin/python" ]; then
  PYTHON="./env/bin/python"
fi

ARGS=("src/phase49_submission_closure.py")
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-DryRun" ]; then
  ARGS+=("--dry-run")
fi

"$PYTHON" "${ARGS[@]}"
