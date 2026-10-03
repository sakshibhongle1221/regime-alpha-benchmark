#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
SYMBOLS=("BTCUSDT" "ETHUSDT")
CREATE_ENV=0
ARCHIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode|-m)
      MODE="$2"
      shift 2
      ;;
    --symbols)
      shift
      SYMBOLS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SYMBOLS+=("$1")
        shift
      done
      ;;
    --create-env)
      CREATE_ENV=1
      shift
      ;;
    --archive)
      ARCHIVE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$CREATE_ENV" -eq 1 && ! -x "env/bin/python" ]]; then
  echo "Creating Python 3.11 virtual environment in env/"
  python3.11 -m venv env
fi

if [[ -x "env/bin/python" ]]; then
  PYTHON="env/bin/python"
else
  PYTHON="python"
fi

echo "Using Python: $PYTHON"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $name"
  "$PYTHON" "$@"
}

if [[ "$CREATE_ENV" -eq 1 ]]; then
  run_step "Install research dependencies" -m pip install -r requirements-research.txt
fi

case "$MODE" in
  dashboard)
    run_step "Launch Streamlit dashboard" -m streamlit run streamlit_app.py
    ;;
  smoke)
    run_step "Compile Python sources" -m compileall src dashboard.py streamlit_app.py
    run_step "Test Phase 39 fold-local safeguards" -m unittest discover -s tests -p "test_*.py" -v
    run_step "Verify or initialize paper artifacts" src/paper_skeleton.py
    run_step "Run validation audit" src/validation_audit.py --symbols "${SYMBOLS[@]}"
    echo
    echo "Smoke reproduction complete."
    ;;
  full)
    run_step "Data health check" src/check.py
    run_step "Generate targets" src/targets.py --symbols "${SYMBOLS[@]}"
    run_step "Train vanilla contrastive encoder" src/train_encoder.py --symbols "${SYMBOLS[@]}"
    run_step "Visualize dense regimes" src/visualize_regimes.py --symbols "${SYMBOLS[@]}"
    run_step "Run classical baselines" src/baselines.py --symbols "${SYMBOLS[@]}"
    run_step "Run alpha models" src/alpha_models.py --symbols "${SYMBOLS[@]}"
    run_step "Regime stability diagnostics" src/regime_stability.py --symbols "${SYMBOLS[@]}"
    run_step "Regime quality diagnostics" src/regime_quality.py --symbols "${SYMBOLS[@]}"
    run_step "Compute budget refresh" src/compute_plan.py --symbols "${SYMBOLS[@]}"
    run_step "Train HMM-guided encoder" src/guided_encoder.py --symbols "${SYMBOLS[@]}" --epochs 30
    run_step "Run time-frequency guided prototype" src/guided_encoder.py --symbols "${SYMBOLS[@]}" --augmentation time_frequency --epochs 3
    run_step "Run fold-local regime benchmark" src/walkforward_regimes.py --symbols "${SYMBOLS[@]}"
    run_step "Run robustness matrix" src/robustness.py
    run_step "Run stress robustness" src/robustness_stress.py
    run_step "Run statistical tests" src/statistical_tests.py
    run_step "Run interpretability" src/interpretability.py --symbols "${SYMBOLS[@]}"
    run_step "Run ablation suite" src/ablation_suite.py
    run_step "Run paper claim tests" src/paper_claim_tests.py
    run_step "Verify or initialize paper artifacts" src/paper_skeleton.py
    run_step "Run validation audit" src/validation_audit.py --symbols "${SYMBOLS[@]}"
    run_step "Build backtest dashboard artifacts" src/backtest.py
    run_step "Compile Python sources" -m compileall src dashboard.py streamlit_app.py

    if [[ "$ARCHIVE" -eq 1 ]]; then
      RUN_ID="local_phase28_reproduction_$(date +%Y%m%d_%H%M%S)"
      run_step "Archive curated run" src/archive_run.py \
        --phase phase28_reproduction \
        --run-id "$RUN_ID" \
        --source-ref HEAD \
        --notes "Local Phase 28 full reproduction run."
    fi

    echo
    echo "Full reproduction complete."
    ;;
  *)
    echo "Invalid mode: $MODE. Use smoke, full, or dashboard." >&2
    exit 2
    ;;
esac
