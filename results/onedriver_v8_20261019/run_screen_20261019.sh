#!/bin/bash
# Screening mono-piloto v8, 8 GP x 5 algoritmos x 5 semillas (2026-10-19). Copia de
# results/onedriver_v7_20261015/run_screen_20261015.sh con el arbol rl_agents_onedriver_v8_20261019 y el wrapper
# scripts/one_driver_v8_20261019.py (via screen_lap0_20261019.py, que ademas anota la accion de la vuelta 0 en
# lap0_seedXX.tsv). Las 5 semillas en paralelo (una por proceso).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v8_20261019
AG=trained_models/rl_agents_onedriver_v8_20261019
mkdir -p $OUT/logs
for s in 01 02 03 04 05; do
  ( LAP0_LOG=$OUT/lap0_seed$s.tsv ~/.venvs/f1-drl-v2/bin/python -u $OUT/screen_lap0_20261019.py \
      --agents-dir $AG/seed$s --models-dir $AG/seed$s/one_driver \
      --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
      > $OUT/logs/screen_seed$s.log 2>&1
    echo "seed$s rc=$?" ) &
done
wait
