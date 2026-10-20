#!/bin/zsh
# 2026-10-11: diagnostico Hungarian v5 (scripts/diag_hungarian_v5_20261011.py), 4 mundos x politicas.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 F1_SIMULATION_DIR=trained_models/simulation_v5_20261010 \
       F1_RL_AGENTS_DIR=trained_models/rl_agents_v5_20261010/selected
POL=(PPO RecurrentPPO QMIP fix:S-23:S-46:M fix:S-21:M-45:S fix:S-34:M fix:S-23:M-46:M)
for r in stoch real; do for y in stoch none; do echo "$r $y"; done; done | \
  xargs -P 4 -L 1 zsh -c 'r=$0; y=$1; ~/.venvs/f1-drl-v2/bin/python scripts/diag_hungarian_v5_20261011.py --rivals $r --yf $y --n 300 --policies '"${POL[*]}"' > results/hungarian_diag_v5_20261011/logs/r${r}_y${y}.log 2>&1; echo "r$r y$y exit $?" >> results/hungarian_diag_v5_20261011/logs/status.txt'
echo ALLDONE >> results/hungarian_diag_v5_20261011/logs/status.txt
