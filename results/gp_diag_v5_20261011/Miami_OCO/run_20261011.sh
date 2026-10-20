#!/bin/zsh
# 2026-10-11: diagnostico Miami v5 en el mundo 2024 (scripts/diag_gp_v5_20261011.py), 6 mundos x politicas.
# Planes fijos: DRL (M->H v22), QMIP (S->M v22), y las mismas paradas bajo el SC real (v28).
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 F1_SIMULATION_DIR=trained_models/simulation_v5_20261010 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5_20261010/selected
D=results/gp_diag_v5_20261011/Miami_OCO
POL=(PPO RecurrentPPO QMIP fix:M-22:H fix:S-22:M fix:M-28:H fix:S-28:M fix:S-22:H)
for w in "stoch stoch" "stoch real" "stoch none" "real real" "real none" "real stoch"; do echo $w; done | \
  xargs -P 3 -L 1 zsh -c 'r=$0; y=$1; ~/.venvs/f1-drl-v2/bin/python scripts/diag_gp_v5_20261011.py --gp Miami_OCO --world 2024 --rivals $r --yf $y --n 300 --policies '"${POL[*]}"' > '$D'/logs/r${r}_y${y}.log 2>&1; echo "r$r y$y exit $?" >> '$D'/logs/status.txt'
echo ALLDONE >> $D/logs/status.txt
