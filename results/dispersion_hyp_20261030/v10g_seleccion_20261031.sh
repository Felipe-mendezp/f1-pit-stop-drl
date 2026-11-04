#!/bin/bash
# 2026-10-31 §14.15 paso 2: seleccion de semillas del arbol v10g (PPO ent 0.05, 8 GP x 10 semillas).
# Identico a results/miami_lap6_diag_20261029/ent05_seleccion_20261030.sh, salvo que el envoltorio
# scripts/eq2_noise_20261030.py va primero (mundo v10g: ruido de la Ec. 2, sigma 0.28, KEEP_STD=1).
#   bash results/dispersion_hyp_20261030/v10g_seleccion_20261031.sh <arbol> <algo> <etiqueta> ["GP1 GP2 ..."]
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
WRAP="scripts/eq2_noise_20261030.py scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py"
GPS="${4:-Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG}"
export F1_SELECT_TREE="$PWD/$1" F1_SELECT_OUT="$PWD/results/seed_selection_v5_20261010/v10f_$3"
mkdir -p "$F1_SELECT_OUT"
$PY $WRAP scripts/select_seeds_v5_20261010.py eval --calib v10 --gps $GPS --algos $2 --n 200 --jobs 8 > "$F1_SELECT_OUT/select_eval.log" 2>&1
$PY $WRAP scripts/select_seeds_v5_20261010.py select --calib v10 --gps $GPS --algos $2 --ok-neutral --first-after 5 --late 7 \
    --min-gap 8 --max-dsq 0 > "$F1_SELECT_OUT/select_pick.log" 2>&1
tail -12 "$F1_SELECT_OUT/select_pick.log"
