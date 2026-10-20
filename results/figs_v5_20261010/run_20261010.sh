#!/bin/zsh
# 2026-10-10: Figs. 6 y 7 v5 (simulador simulation_v5_20261010, agentes v5 seleccionados).
# Usa scripts/evaluate_world2024_20261010.py --world none, que reproduce exactamente las Figs. 6-7 v4
# cuando apunta al arbol v4 (verificado). Correr despues de select_seeds_v5_20261010.py select.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_SIMULATION_DIR=trained_models/simulation_v5_20261010 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5_20261010/selected
GPS=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_SAI Belgian_NOR Dutch_RUS Singapore_HUL United_States_PIA)
for f in 6 7; do for g in $GPS; do echo "$f $g"; done; done | \
  xargs -P 8 -L 1 zsh -c 'f=$0; g=$1; ~/.venvs/f1-drl-v2/bin/python scripts/evaluate_world2024_20261010.py --fig $f --world none --gps $g --out results/figs_v5_20261010 > results/figs_v5_20261010/logs/fig${f}_$g.log 2>&1; echo "fig$f $g exit $?" >> results/figs_v5_20261010/logs/status.txt'
echo ALLDONE >> results/figs_v5_20261010/logs/status.txt
