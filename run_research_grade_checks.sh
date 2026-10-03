#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-artifact}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" src/research_grade_checks.py --mode "${MODE}"
