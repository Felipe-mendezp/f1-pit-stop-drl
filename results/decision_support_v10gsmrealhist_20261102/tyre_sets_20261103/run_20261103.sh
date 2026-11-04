#!/bin/zsh
# 2026-11-03 — Juegos disponibles en los puntos de decision de la Fig. 11 (tabla de apendice, idea del autor).
# Mismo entorno que fig11_20261103.sh. Salida: <este folder>/<gp>.json
#   zsh results/decision_support_v10gsmrealhist_20261102/tyre_sets_20261103/run_20261103.sh
cd "$(dirname $0)/../../.."
OUT=results/decision_support_v10gsmrealhist_20261102/tyre_sets_20261103
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
export F1_DS_FORCE=hist F1_DS_WORLD2024=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_startmix_20261101/selected
unset F1_RIVAL_DNF F1_START_MIX
run () {  # <gp> <driver> <model-dir> <laps> <tag>
  ~/.venvs/f1-drl-v2/bin/python scripts/gp_configs_fix_20261028.py scripts/ds_rollouts_realhist_20261024.py --run \
      scripts/tyre_sets_at_decisions_20261103.py --gp "$1" --driver "$2" --model-dir "$3" --laps "$4" \
      --algo PPO --following-gap-scale 0.26 --out "$OUT/$5.json"
  echo "rc=$? $5"
}
run "Bahrain Grand Prix"        Driver_ALO Bahrain_ALO       15,39   bahrain_alo
run "Saudi Arabian Grand Prix"  Driver_NOR Saudi_Arabian_NOR 7,29,34 saudi_arabian_nor
run "Miami Grand Prix"          Driver_OCO Miami_OCO         22      miami_oco
run "Hungarian Grand Prix"      Driver_TSU Hungarian_TSU     29      hungarian_tsu
run "Belgian Grand Prix"        Driver_GAS Belgian_GAS       9,25,26 belgian_gas
run "Dutch Grand Prix"          Driver_STR Dutch_STR         30      dutch_str
run "Singapore Grand Prix"      Driver_HUL Singapore_HUL     29      singapore_hul
run "United States Grand Prix"  Driver_MAG United_States_MAG 3,17,36 united_states_mag
