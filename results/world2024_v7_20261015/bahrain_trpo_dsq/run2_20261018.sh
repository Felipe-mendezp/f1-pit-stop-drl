#!/bin/zsh
# 2026-10-18 (2): trazas cortas (100 carreras) con los compuestos de partida de los rivales: semilla 5 Fig. 6 y 7,
# y semillas 3, 4, 7, 9 (las otras que pasan el filtro 5/7/8) y 10 en la Fig. 7.
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
PY=~/.venvs/f1-drl-v2/bin/python; D=results/world2024_v7_20261015/bahrain_trpo_dsq; A=trained_models/rl_agents_v7_20261015
run() { local n=$1 ag=$2; shift 2
  F1_RL_AGENTS_DIR=$ag $PY scripts/gp_drivers_v5hd_20261014.py $D/diag_20261018.py $@ --world 2024 --gps Bahrain_ALO \
    --n-sims 100 --algos TRPO --out $D/short/$n > $D/short/$n.log 2>&1
  echo "$(date +%T) short/$n exit $?" >> $D/status.txt }
mkdir -p $D/short
run s05_f6 $A/seed05 --fig 6 --real-dnf &
for s in 03 04 05 07 09 10; do run s${s}_f7 $A/seed$s --fig 7 & done
for s in 03 04 07; do run s${s}_f6 $A/seed$s --fig 6 --real-dnf & done
wait; echo "$(date +%T) fin2" >> $D/status.txt
