#!/bin/zsh
# 2026-10-28 (arbol v10c: compuestos corregidos con OpenF1 y pilotos v10c): barrido de apoyo a la decision v10 (simulador v10, agentes retrained_v10c_20261028/selected; copia de
# run_20261019.sh, con los agentes v9r2 y los juegos de neumaticos 2024 reutilizables).
#   zsh results/decision_support_v10cw24belief_20261028/run_20261028.sh [args extra, p. ej. --gps Saudi_Arabian_NOR --M 20]
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10c F1_VSC_FIRST_LAP_ONLY=1 F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10c_20261028
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF
~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v10 --M 500 --workers 8 --figures "$@"
