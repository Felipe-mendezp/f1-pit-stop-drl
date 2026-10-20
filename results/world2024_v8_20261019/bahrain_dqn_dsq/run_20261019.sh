#!/bin/zsh
# 2026-10-19: diagnostico Bahrain DQN en la Fig. 7 v8 (11.6 % descalificado; semilla 8 elegida con reglas 5/7/8).
# Mismo protocolo que results/world2024_v7_20261015/bahrain_trpo_dsq/run_20261018.sh, con las variables v8 de
# reproduce/pipeline_v8_20261019.sh y los agentes rl_agents_v8_20261018. trace_s08_f7 / trace_s08_f6: traza
# (results/world2024_v7_20261015/bahrain_trpo_dsq/diag_20261018.py, sin cambios); seeds/seedXX: Fig. 7 Bahrain con
# cada semilla de DQN (solo DQN + QMIP, 500 carreras).
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
PY=~/.venvs/f1-drl-v2/bin/python; D=results/world2024_v8_20261019/bahrain_dqn_dsq; A=trained_models/rl_agents_v8_20261018
G=(scripts/gp_drivers_v5hd_20261014.py); DIAG=results/world2024_v7_20261015/bahrain_trpo_dsq/diag_20261018.py
mkdir -p $D/seeds
run() { # $1 nombre, $2 dir de agentes, resto args
  local n=$1 ag=$2; shift 2
  F1_RL_AGENTS_DIR=$ag $PY $G $@ --world 2024 --gps Bahrain_ALO --n-sims 500 --algos DQN --out $D/$n > $D/$n.log 2>&1
  echo "$(date +%T) $n exit $?" >> $D/status.txt
}
run trace_s08_f7 $A/seed08 $DIAG --fig 7 &
run trace_s08_f6 $A/seed08 $DIAG --fig 6 --real-dnf &
for s in 01 02 03 04 05 06; do run seeds/seed$s $A/seed$s scripts/evaluate_world2024_20261010.py --fig 7 & done
wait
for s in 07 09 10; do run seeds/seed$s $A/seed$s scripts/evaluate_world2024_20261010.py --fig 7 & done
wait
echo "$(date +%T) fin" >> $D/status.txt
