#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MAX_FOLDS="${1:-0}"
RUN_NAME="${2:-phase39r_classical_full}"
OUTPUT_DIR="${3:-models}"
OUTPUT_PREFIX="${4:-crypto20_repaired_classical_}"
RESUME="${5:-0}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ARGS=(
  src/repaired_classical_baseline.py
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
)
if [[ "${MAX_FOLDS}" -gt 0 ]]; then
  ARGS+=(--max-folds "${MAX_FOLDS}")
fi
if [[ "${RESUME}" -eq 1 ]]; then
  ARGS+=(--resume)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
