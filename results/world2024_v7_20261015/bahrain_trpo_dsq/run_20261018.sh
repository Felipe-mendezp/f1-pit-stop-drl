#!/bin/zsh
# 2026-10-18: diagnostico Bahrain TRPO. trace_s05_f7 / trace_s01_f7 / trace_s05_f6: traza (diag_20261018.py);
# seeds/seedXX: Fig. 7 Bahrain con cada semilla de TRPO (solo TRPO + QMIP, 500 carreras, semilla 0 como la Fig. 7).
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
PY=~/.venvs/f1-drl-v2/bin/python; D=results/world2024_v7_20261015/bahrain_trpo_dsq; A=trained_models/rl_agents_v7_20261015
G=(scripts/gp_drivers_v5hd_20261014.py)
run() { # $1 nombre, $2 dir de agentes, resto args
  local n=$1 ag=$2; shift 2
  F1_RL_AGENTS_DIR=$ag $PY $G $@ --world 2024 --gps Bahrain_ALO --n-sims 500 --algos TRPO --out $D/$n > $D/$n.log 2>&1
  echo "$(date +%T) $n exit $?" >> $D/status.txt
}
run trace_s05_f7 $A/seed05 $D/diag_20261018.py --fig 7 &
run trace_s01_f7 $A/seed01 $D/diag_20261018.py --fig 7 &
run trace_s05_f6 $A/seed05 $D/diag_20261018.py --fig 6 --real-dnf &
for s in 02 03 04 06 07 08; do run seeds/seed$s $A/seed$s scripts/evaluate_world2024_20261010.py --fig 7 & done
wait
for s in 09 10; do run seeds/seed$s $A/seed$s scripts/evaluate_world2024_20261010.py --fig 7 & done
wait
echo "$(date +%T) fin" >> $D/status.txt
