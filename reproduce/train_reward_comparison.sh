#!/usr/bin/env bash
# Retrain the reward-comparison agents of the paper from reproduce/MODELS_MANIFEST.csv into a NEW tree
# (default trained_models/retrained_reward_comparison). Extra arguments are passed to run_manifest.py,
# e.g.  bash reproduce/train_reward_comparison.sh --dry-run    or    --only Saudi
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON="${PYTHON:-python}"
OUT="${OUT:-trained_models/retrained_reward_comparison}"
exec "$PYTHON" reproduce/run_manifest.py --scope reward_comparison --out "$OUT" "$@"
