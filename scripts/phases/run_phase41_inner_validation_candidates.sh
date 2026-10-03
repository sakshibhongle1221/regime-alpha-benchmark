#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_FOLDS="${MAX_FOLDS:-1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-phase41_classical_}"
REPORT_PATH="${REPORT_PATH:-reports/phase41_inner_validation_candidate_run.md}"
METHODS="${METHODS:-global_lgbm regime_lgbm_hmm regime_lgbm_kmeans regime_lgbm_vol_bucket}"

"${PYTHON_BIN}" src/phase41_inner_validation_candidates.py \
  --universe crypto20 \
  --max-folds "${MAX_FOLDS}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --report-path "${REPORT_PATH}" \
  --methods ${METHODS}
