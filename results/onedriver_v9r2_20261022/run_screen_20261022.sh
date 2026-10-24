#!/bin/bash
# Screening mono-piloto v9r2 (juegos reutilizables, F1_TYRE_REUSE=1), 8 GP x 5 algoritmos x 5 semillas (2026-10-22; copia de run_screen_20261021.sh v9r). Copia de
# results/onedriver_v8_20261019/run_screen_20261019.sh con el arbol rl_agents_onedriver_v9r2_20261022 y
# screen_v9r2_20261022.py (juegos de neumaticos por reglamento; anota vuelta 0 y pedidos sin juegos en events_seedXX.tsv).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v9r2_20261022
AG=trained_models/rl_agents_onedriver_v9r2_20261022
mkdir -p $OUT/logs
for s in 01 02 03 04 05; do
  ( LOG_FILE=$OUT/events_seed$s.tsv ~/.venvs/f1-drl-v2/bin/python -u $OUT/screen_v9r2_20261022.py \
      --agents-dir $AG/seed$s --models-dir $AG/seed$s/one_driver \
      --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
      > $OUT/logs/screen_seed$s.log 2>&1
    echo "seed$s rc=$?" ) &
done
wait
