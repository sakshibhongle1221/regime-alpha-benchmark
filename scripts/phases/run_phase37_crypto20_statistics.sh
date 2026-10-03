#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BOOTSTRAP_SAMPLES="${1:-10000}"
DM_LAG="${2:-7}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" src/statistical_tests.py \
  --predictions models/crypto20_walkforward_alpha_oos_predictions.csv \
  --experiment-results models/crypto20_walkforward_experiment_results.csv \
  --output-prefix crypto20_ \
  --reference-methods global_lgbm regime_lgbm_hmm regime_lgbm_kmeans \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --dm-lag "${DM_LAG}"
