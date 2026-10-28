#!/bin/zsh
# 2026-10-27: chequeo (b) de la regla de paradas bajo SC (scripts/ds_sc_second_lap_check_20261027.py).
#   zsh results/sc_second_lap_check_20261027/run_20261027.sh <fig8|fig11> <GP> <Driver_X> <ModelDir> <lap> <k> <M> <tag>
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh >/dev/null 2>&1
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 OMP_NUM_THREADS=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF F1_START_MIX
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015 F1_DS_WORLD2024=1
if [[ $1 == fig8 ]]; then
  export F1_RL_AGENTS_DIR=$PWD/trained_models/rl_agents_v9r2_20261022/selected F1_SC2_BASE=belief; unset F1_DS_FORCE
else
  export F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v9r2_startmix_20261025/selected F1_SC2_BASE=realhist F1_DS_FORCE=hist
fi
export F1_SC_PIT_MAX_IDX=$6
O=results/sc_second_lap_check_20261027/$1
mkdir -p $O
~/.venvs/f1-drl-v2/bin/python scripts/gp_drivers_v5hd_20261014.py scripts/ds_sc_second_lap_check_20261027.py \
  --gp "$2" --driver $3 --model-dir $4 --algo PPO --M $7 --fast-env --legal-mask --real-2024 --sim-future \
  --following-gap-scale 1.0 --out-dir $O --laps $5 --tag-suffix _$8
