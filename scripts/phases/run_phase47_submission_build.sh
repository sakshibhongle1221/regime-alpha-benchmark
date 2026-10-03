#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
ARGS=("src/phase47_submission_build.py")

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON}" "${ARGS[@]}"
