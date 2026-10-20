#!/bin/zsh
# 2026-10-11: prueba world-2024 sobre v5. Agentes v5 seleccionados y QMIP creen el simulador v5
# (neumaticos opcion3 2021-23 + coef. de piloto Ec. 2); el mundo usa neumaticos opcion3 2021-24.
# Mismo envoltorio que results/world2024_20261010 (v4). Comparar con results/figs_v5_20261010 (--world none).
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_SIMULATION_DIR=trained_models/simulation_v5_20261010 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5_20261010/selected
GPS=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_SAI Belgian_NOR Dutch_RUS Singapore_HUL United_States_PIA)
for f in 6 7; do for g in $GPS; do echo "$f $g"; done; done | \
  xargs -P 8 -L 1 zsh -c 'f=$0; g=$1; ~/.venvs/f1-drl-v2/bin/python scripts/evaluate_world2024_20261010.py --fig $f --world 2024 --gps $g --out results/world2024_v5_20261011 > results/world2024_v5_20261011/logs/fig${f}_$g.log 2>&1; echo "fig$f $g exit $?" >> results/world2024_v5_20261011/logs/status.txt'
echo ALLDONE >> results/world2024_v5_20261011/logs/status.txt
