#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EPOCHS="${1:-30}"
BATCH_SIZE="${2:-128}"
MAX_WINDOWS="${3:-5000}"
MAX_FOLDS="${4:-0}"
RUN_NAME="${5:-phase39r_neural_full_v1}"
OUTPUT_DIR="${6:-models}"
OUTPUT_PREFIX="${7:-crypto20_repaired_fold_local_}"
REPORT_PATH="${8:-reports/phase39r_neural_fold_local_results.md}"
RESUME="${9:-0}"
CALENDAR_AUDIT_ONLY="${10:-0}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ARGS=(
  src/fold_local_encoder_walkforward.py
  --universe crypto20
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --max-windows "${MAX_WINDOWS}"
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --report-path "${REPORT_PATH}"
)

if [[ "${MAX_FOLDS}" -gt 0 ]]; then
  ARGS+=(--max-folds "${MAX_FOLDS}")
fi

if [[ "${RESUME}" -eq 1 ]]; then
  ARGS+=(--resume)
fi

if [[ "${CALENDAR_AUDIT_ONLY}" -eq 1 ]]; then
  ARGS+=(--calendar-audit-only)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
