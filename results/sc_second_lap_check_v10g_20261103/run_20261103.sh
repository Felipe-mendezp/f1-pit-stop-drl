#!/bin/zsh
# 2026-11-03: chequeo de la parada en la 2.a vuelta del SC en v10g (Fig. 8, Saudi NOR v. 7; supplementary.tex, "Three remarks").
# Copia de ../sc_second_lap_check_20261027/run_20261027.sh con el entorno de ../decision_support_v10gw24belief_20261103/run_20261103.sh.
#   zsh results/sc_second_lap_check_v10g_20261103/run_20261103.sh <k> <M> <tag>     k=1 control (debe coincidir con el barrido), k=2 chequeo
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh >/dev/null 2>&1
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1 OMP_NUM_THREADS=1 F1_DS_WORLD2024=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028 F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_20261030/selected
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
unset F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
export F1_SC2_BASE=belief F1_SC_PIT_MAX_IDX=$1
O=results/sc_second_lap_check_v10g_20261103/fig8
~/.venvs/f1-drl-v2/bin/python scripts/gp_configs_fix_20261028.py scripts/ds_sc_second_lap_check_20261027.py \
  --gp "Saudi Arabian Grand Prix" --driver Driver_NOR --model-dir Saudi_Arabian_NOR --algo PPO --M $2 --fast-env --legal-mask --real-2024 --sim-future \
  --following-gap-scale 0.26 --out-dir $O --laps 7 --tag-suffix _$3
