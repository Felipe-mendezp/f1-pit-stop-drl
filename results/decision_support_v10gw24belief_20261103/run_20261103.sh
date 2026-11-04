#!/bin/zsh
# 2026-11-03 — PENDIENTE DE CORRER (decision del autor: dejarlo listo y lanzarlo despues).
# Barrido de apoyo a la decision en v10g, solo PPO: regenera los datos de las Figs. 8-10, que siguen siendo del
# 24-oct (mundo v9r2) mientras el resto del paper ya esta en v10g. Copia de
# results/decision_support_v10eGQfw24belief_20261029/run_20261029.sh mas el ruido de la Ec. 2 (v10g) y
# --version v10g (agentes trained_models/retrained_v10g_20261030/selected, los seleccionados el 31-oct).
# Son 16 corridas (8 GP x 2 escenarios), unas 2 h en el Mac con 8 workers; usar caffeinate.
#   caffeinate -i zsh results/decision_support_v10gw24belief_20261103/run_20261103.sh 2>&1 | tee results/decision_support_v10gw24belief_20261103/sweep_20261103.log
# Solo el caso de las Figs. 8-10 (unos 35 min):  ... run_20261103.sh --gps Saudi_Arabian_NOR
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
unset F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v10g --algos PPO --M 500 --workers 8 --figures "$@"
echo "rc=$?"
