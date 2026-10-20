#!/bin/zsh
# 2026-10-12: Figs. 6/7 en el mundo 2024 con la opcion 3 jerarquica (version del paper).
# Agentes v5h seleccionados (5 DRL) y QMIP creen el simulador v5h8 (opcion3h 2021-23 + coef. de piloto Ec. 2);
# el mundo usa neumaticos opcion3h 2021-24 (results/tyre_coef_study_20260930/world2024h_20261012).
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5h_20261011/selected
GPS=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_SAI Belgian_NOR Dutch_RUS Singapore_HUL United_States_PIA)
for f in 6 7; do for g in $GPS; do echo "$f $g"; done; done | \
  xargs -P 8 -L 1 zsh -c 'f=$0; g=$1; ~/.venvs/f1-drl-v2/bin/python scripts/evaluate_world2024_20261010.py --fig $f --world 2024 --gps $g --algos DQN A2C TRPO PPO RecurrentPPO --out results/world2024_v5h_20261012 > results/world2024_v5h_20261012/logs/fig${f}_$g.log 2>&1; echo "fig$f $g exit $?" >> results/world2024_v5h_20261012/logs/status.txt'
echo ALLDONE >> results/world2024_v5h_20261012/logs/status.txt
