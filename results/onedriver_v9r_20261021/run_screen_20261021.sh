#!/bin/bash
# Screening mono-piloto v9r, 8 GP x 5 algoritmos x 5 semillas (2026-10-21). Copia de
# results/onedriver_v8_20261019/run_screen_20261019.sh con el arbol rl_agents_onedriver_v9r_20261021 y
# screen_v9r_20261021.py (juegos de neumaticos por reglamento; anota vuelta 0 y pedidos sin juegos en events_seedXX.tsv).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v9r_20261021
AG=trained_models/rl_agents_onedriver_v9r_20261021
mkdir -p $OUT/logs
for s in 01 02 03 04 05; do
  ( LOG_FILE=$OUT/events_seed$s.tsv ~/.venvs/f1-drl-v2/bin/python -u $OUT/screen_v9r_20261021.py \
      --agents-dir $AG/seed$s --models-dir $AG/seed$s/one_driver \
      --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
      > $OUT/logs/screen_seed$s.log 2>&1
    echo "seed$s rc=$?" ) &
done
wait
