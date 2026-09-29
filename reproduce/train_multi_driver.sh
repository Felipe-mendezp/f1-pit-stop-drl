#!/usr/bin/env bash
# Retrain the multi-driver agents of the paper from reproduce/MODELS_MANIFEST.csv into a NEW tree
# (default trained_models/retrained_multi_driver). Extra arguments are passed to run_manifest.py,
# e.g.  bash reproduce/train_multi_driver.sh --dry-run    or    --only Saudi
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON="${PYTHON:-python}"
OUT="${OUT:-trained_models/retrained_multi_driver}"
exec "$PYTHON" reproduce/run_manifest.py --scope multi_driver --out "$OUT" "$@"
