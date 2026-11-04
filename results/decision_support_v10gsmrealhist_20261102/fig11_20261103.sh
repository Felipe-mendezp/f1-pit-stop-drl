#!/bin/zsh
# 2026-11-03 — Fig. 11 (fig:realhist) con el barrido B de los agentes v10gsm. Mismo entorno que run_20261102.sh mas
# F1_DS_WORLD2024=1 F1_PAPER_FONTS=1 y los arboles del simulador y de los agentes, como pide
# scripts/plot_realhist_stints_20261026.py. Salida: figs/realhist_stints_grid4x2.{pdf,png} (la que usa el paper).
#   zsh results/decision_support_v10gsmrealhist_20261102/fig11_20261103.sh
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
export F1_DS_FORCE=hist F1_DS_WORLD2024=1 F1_PAPER_FONTS=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_startmix_20261101/selected
unset F1_RIVAL_DNF F1_START_MIX
~/.venvs/f1-drl-v2/bin/python scripts/plot_realhist_stints_20261026.py \
    --res results/decision_support_v10gsmrealhist_20261102 --laps compare_laps_20261103.csv
echo "rc=$?"
