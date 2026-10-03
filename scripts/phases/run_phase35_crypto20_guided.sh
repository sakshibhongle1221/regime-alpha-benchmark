#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EPOCHS=30
BATCH_SIZE=128
MAX_WINDOWS=0
TRAIN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-windows)
      MAX_WINDOWS="$2"
      shift 2
      ;;
    --train-only)
      TRAIN_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done


if [[ -x "env/bin/python" ]]; then
  PYTHON="env/bin/python"
else
  PYTHON="python"
fi

if [[ ! -f "models/crypto20_regime_assignments.csv" ]]; then
  echo "Missing models/crypto20_regime_assignments.csv. Run src/crypto20_regime_benchmark.py --universe crypto20 first." >&2
  exit 1
fi

mkdir -p .tmp
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH=".tmp/phase35_crypto20_guided_${TIMESTAMP}.log"

ARGS=(
  "src/guided_encoder.py"
  "--universe" "crypto20"
  "--eligible-only"
  "--hmm-assignment-path" "models/crypto20_regime_assignments.csv"
  "--output-prefix" "crypto20_guided_encoder"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
)

if [[ "$MAX_WINDOWS" -gt 0 ]]; then
  ARGS+=("--max-windows" "$MAX_WINDOWS")
fi

if [[ "$TRAIN_ONLY" -eq 1 ]]; then
  ARGS+=("--train-only")
fi

echo "Starting Phase 35 Crypto-20 guided encoder run..."
echo "Log: $LOG_PATH"
echo "Command: $PYTHON ${ARGS[*]}"

"$PYTHON" "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"

echo "Phase 35 command finished. Log saved to $LOG_PATH"
