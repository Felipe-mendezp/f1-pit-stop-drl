#!/bin/bash
# Screening mono-piloto v10 (2026-10-27; copia de results/onedriver_v9r2_20261022/run_screen_20261022.sh con el arbol de agentes retrained_v10_onedriver_20261027 y screen_v10_20261027.py).
# results/onedriver_v8_20261019/run_screen_20261019.sh con el arbol retrained_v10_onedriver_20261027 y
# screen_v10_20261027.py (juegos de neumaticos por reglamento; anota vuelta 0 y pedidos sin juegos en events_seedXX.tsv).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v10_20261027
AG=trained_models/retrained_v10_onedriver_20261027
mkdir -p $OUT/logs
for s in 01 02 03 04 05; do
  ( LOG_FILE=$OUT/events_seed$s.tsv ~/.venvs/f1-drl-v2/bin/python -u $OUT/screen_v10_20261027.py \
      --agents-dir $AG/seed$s --models-dir $AG/seed$s/one_driver \
      --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
      > $OUT/logs/screen_seed$s.log 2>&1
    echo "seed$s rc=$?" ) &
done
wait
