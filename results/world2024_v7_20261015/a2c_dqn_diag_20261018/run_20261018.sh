#!/bin/zsh
# 2026-10-18: diagnostico Dutch A2C / US A2C (descalificaciones Fig. 7) y US DQN (paradas tardias Fig. 6), v7 5/7/8.
# Trazas con ../bahrain_trpo_dsq/diag_20261018.py (200 carreras) para las semillas elegidas (Dutch A2C 9, US A2C 7,
# US DQN 7) en Fig. 6 (abandonos reales) y Fig. 7; y Fig. 7 de A2C con las 10 semillas en Dutch y US (seeds/).
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
PY=~/.venvs/f1-drl-v2/bin/python; D=results/world2024_v7_20261015/a2c_dqn_diag_20261018; A=trained_models/rl_agents_v7_20261015
T=results/world2024_v7_20261015/bahrain_trpo_dsq/diag_20261018.py
run() { # $1 nombre, $2 semilla, $3 GP, $4 algo, resto args
  local n=$1 s=$2 g=$3 al=$4; shift 4; mkdir -p $(dirname $D/$n)
  F1_RL_AGENTS_DIR=$A/seed$s $PY scripts/gp_drivers_v5hd_20261014.py $@ --world 2024 --gps $g --n-sims 200 \
    --algos $al --out $D/$n > $D/$n.log 2>&1
  echo "$(date +%T) $n exit $?" >> $D/status.txt }
run trace/dutch_a2c_s09_f7 09 Dutch_STR A2C $T --fig 7 &
run trace/dutch_a2c_s09_f6 09 Dutch_STR A2C $T --fig 6 --real-dnf &
run trace/us_a2c_s07_f7 07 United_States_MAG A2C $T --fig 7 &
run trace/us_a2c_s07_f6 07 United_States_MAG A2C $T --fig 6 --real-dnf &
run trace/us_dqn_s07_f6 07 United_States_MAG DQN $T --fig 6 --real-dnf &
run trace/us_dqn_s07_f7 07 United_States_MAG DQN $T --fig 7 &
wait
for g in Dutch_STR United_States_MAG; do
  for s in 01 02 03 04 05; do run seeds/${g}_a2c_s$s $s $g A2C scripts/evaluate_world2024_20261010.py --fig 7 & done; wait
  for s in 06 07 08 09 10; do run seeds/${g}_a2c_s$s $s $g A2C scripts/evaluate_world2024_20261010.py --fig 7 & done; wait
done
echo "$(date +%T) fin" >> $D/status.txt
