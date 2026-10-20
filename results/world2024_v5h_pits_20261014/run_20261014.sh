#!/bin/zsh
# 2026-10-14: Figs. 6/7 v5h (mundo 2024) de los 4 GP que no cambian de piloto, igual que
# results/world2024_v5h_20261012/run_20261012.sh, para tener el conteo real de paradas (columnas n_pits_all y pit_laps
# que agrega scripts/evaluate_world2024_20261010.py desde el 2026-10-14; D28). Las posiciones deben ser identicas.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5h_20261011/selected
OUT=results/world2024_v5h_pits_20261014
GPS=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Singapore_HUL)
for f in 6 7; do for g in $GPS; do echo "$f $g"; done; done | \
  xargs -P 8 -L 1 zsh -c 'f=$0; g=$1; ~/.venvs/f1-drl-v2/bin/python scripts/evaluate_world2024_20261010.py --fig $f --world 2024 --gps $g --algos DQN A2C TRPO PPO RecurrentPPO --out results/world2024_v5h_pits_20261014 > results/world2024_v5h_pits_20261014/logs/fig${f}_$g.log 2>&1; echo "$(date +%T) fig$f $g exit $?" >> results/world2024_v5h_pits_20261014/logs/status.txt'
echo "$(date +%T) ALLDONE" >> $OUT/logs/status.txt
