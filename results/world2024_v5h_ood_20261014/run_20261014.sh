#!/bin/zsh
# 2026-10-14: intervenciones DNF (intervene_20261014.py) en la Fig. 7 v5h, 200 carreras (= primeras 200 semillas de
# results/world2024_v5h_pits_20261014/fig7_2024), los 3 GP con abandonos y paradas raras.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5h_20261011/selected
D=results/world2024_v5h_ood_20261014
for m in nodnf dnfobs; do for g in Saudi_Arabian_NOR Miami_OCO Singapore_HUL; do echo "$m $g"; done; done | \
  xargs -P 6 -L 1 zsh -c 'm=$0; g=$1; ~/.venvs/f1-drl-v2/bin/python results/world2024_v5h_ood_20261014/intervene_20261014.py --mode $m --fig 7 --world 2024 --gps $g --n-sims 200 --algos DQN A2C TRPO PPO RecurrentPPO --out results/world2024_v5h_ood_20261014/$m > results/world2024_v5h_ood_20261014/logs/${m}_$g.log 2>&1; echo "$(date +%T) $m $g exit $?" >> results/world2024_v5h_ood_20261014/logs/status.txt'
echo "$(date +%T) ALLDONE" >> $D/logs/status.txt
