#!/bin/zsh
# 2026-10-19: barrido de apoyo a la decision v8 (mismo protocolo que v7, ver README de decision_support_v7w24belief_20261015).
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
unset F1_RIVAL_DNF
~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v8 --M 500 --workers 8 --figures
