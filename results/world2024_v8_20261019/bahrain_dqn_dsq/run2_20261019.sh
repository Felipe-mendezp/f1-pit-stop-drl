#!/bin/zsh
# 2026-10-19: repite las trazas de run_20261019.sh con diag_dqn_20261019.py (trace.csv quedo vacio con DQN).
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
PY=~/.venvs/f1-drl-v2/bin/python; D=results/world2024_v8_20261019/bahrain_dqn_dsq; A=trained_models/rl_agents_v8_20261018
run() { local n=$1 ag=$2; shift 2
  F1_RL_AGENTS_DIR=$ag $PY scripts/gp_drivers_v5hd_20261014.py $D/diag_dqn_20261019.py $@ --world 2024 --gps Bahrain_ALO \
    --n-sims 500 --algos DQN --out $D/$n > $D/$n.log 2>&1
  echo "$(date +%T) $n exit $?" >> $D/status.txt; }
run trace2_s08_f7 $A/seed08 --fig 7 &
run trace2_s08_f6 $A/seed08 --fig 6 --real-dnf &
run trace2_s03_f7 $A/seed03 --fig 7 &
wait
