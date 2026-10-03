#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=()

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON_BIN}" src/phase44_paper_readiness_package.py "${ARGS[@]}"
