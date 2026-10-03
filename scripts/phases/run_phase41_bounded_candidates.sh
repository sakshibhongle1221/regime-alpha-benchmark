#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" src/phase41_bounded_candidates.py
"${PYTHON_BIN}" -m unittest tests.test_phase41_bounded_candidates -v

echo "OK: Phase 41 bounded candidate protocol artifacts and tests completed."
