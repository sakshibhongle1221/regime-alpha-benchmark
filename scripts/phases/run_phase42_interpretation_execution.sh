#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"

ARGS=("src/phase42_interpretation_execution.py" "--universe" "crypto20")
if [[ "${1:-}" == "--skip-feature-diagnostics" ]]; then
  ARGS+=("--skip-feature-diagnostics")
fi

"${PYTHON}" "${ARGS[@]}"
