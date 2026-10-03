#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MAX_FOLDS="${1:-}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ARGS=(
  "src/walkforward_regimes.py"
  "--universe" "crypto20"
  "--skip-contrastive"
  "--output-prefix" "crypto20_walkforward_"
  "--guided-assignment-path" "models/crypto20_guided_encoder_assignments.csv"
  "--guided-embedding-path" "models/crypto20_guided_encoder_embeddings.npy"
)

if [[ -n "${MAX_FOLDS}" && "${MAX_FOLDS}" != "0" ]]; then
  ARGS+=("--max-folds" "${MAX_FOLDS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
