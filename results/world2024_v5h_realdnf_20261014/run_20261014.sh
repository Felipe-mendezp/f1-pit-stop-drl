#!/bin/zsh
# 2026-10-14: Fig. 6 v5h (mundo 2024) con los abandonos de los rivales fijos a los reales de 2024 (--real-dnf de
# scripts/evaluate_world2024_20261010.py). Mismo protocolo que results/world2024_v5h_20261012/run_20261012.sh
# (semilla 42, 500 carreras, 5 DRL + QMIP); la base sin abandonos es results/world2024_v5h_20261012/fig6_2024.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5h_20261011/selected
OUT=results/world2024_v5h_realdnf_20261014
echo "$(date +%T) start" >> $OUT/logs/status.txt
GPS=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_SAI Belgian_NOR Dutch_RUS Singapore_HUL United_States_PIA)
for g in $GPS; do echo $g; done | \
  xargs -P 8 -L 1 zsh -c 'g=$0; ~/.venvs/f1-drl-v2/bin/python scripts/evaluate_world2024_20261010.py --fig 6 --world 2024 --real-dnf --gps $g --algos DQN A2C TRPO PPO RecurrentPPO --out results/world2024_v5h_realdnf_20261014 > results/world2024_v5h_realdnf_20261014/logs/fig6_$g.log 2>&1; echo "$(date +%T) fig6 $g exit $?" >> results/world2024_v5h_realdnf_20261014/logs/status.txt'
echo "$(date +%T) ALLDONE" >> $OUT/logs/status.txt
