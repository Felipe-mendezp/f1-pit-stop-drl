#!/bin/zsh
# 2026-10-29 (adelantamiento v10f; copia de results/decision_support_v10eGQw24belief_20261028/run_20261028.sh): barrido de
# apoyo a la decision con el arbol simulation_v10eGQ_20261028 (logit v10f) y los agentes retrained_v10eGQf_20261029/selected.
# Agrega las variables de scripts/overtake_v10f_20261029.py (logit en todos los pares sobre el gap medio; gap de
# seguimiento 0.4 + Exp(0.26)); el barrido pasa --following-gap-scale 0.26 con --version v10f.
#   zsh results/decision_support_v10eGQfw24belief_20261029/run_20261029.sh [args extra, p. ej. --gps Saudi_Arabian_NOR --M 20]
# Fig. 11 (post-entrenados): mismo entorno + F1_DS_FORCE=hist y --version v10fsmhist.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1 F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
unset F1_RIVAL_DNF
~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v10f --M 500 --workers 8 --figures "$@"
