#!/bin/bash
# 2026-10-29 B3: PPO Miami OCO con mas entropia (0.05 y 0.10), semillas 1-2, 1.5M pasos, en el Mac. Mismo comando y
# envoltorios que las tareas del cluster (tasks_v10eGQf_20261029.tsv), salvo --ent-coef/--timesteps/--out-root.
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 F1_VEC_DUMMY=1 MPLBACKEND=Agg
unset F1_RL_AGENTS_DIR F1_WORLD2024_TYRES F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
PY=~/.venvs/f1-drl-v2/bin/python
OUT=trained_models/probe_miami_entropy_20261029
mkdir -p $OUT/logs
for ENT in 0.05 0.10; do for S in 1 2; do
  N=ent${ENT}_seed0$S
  $PY scripts/overtake_v10f_20261029.py scripts/tyre_ages_v10eGQ_20261028.py scripts/v10_wrapper_20261027.py \
      scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py \
      scripts/train_ppo_fix_20260820.py --ent-coef $ENT --gps 'Miami Grand Prix' --seeds $S --timesteps 1500000 \
      --fast-env --torch-threads 2 --following-gap-scale 0.26 --sc-bunching --sc-bunch-delta 1.5 \
      --out-root $OUT/$N > $OUT/logs/$N.log 2>&1 &
done; done
wait
echo fin
