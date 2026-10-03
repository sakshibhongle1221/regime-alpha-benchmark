#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/phase43b_locked_holdout_registration_v1.json}"
python src/phase43b_locked_holdout_registration.py --config "$CONFIG"
