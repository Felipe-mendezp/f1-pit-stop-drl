#!/bin/bash
# Screening mono-piloto v7 (sin filtro), 8 GP x 5 algoritmos x 5 semillas (2026-10-15). Mismo protocolo que
# results/onedriver_v5hnf_20261014/run_screen_20261014.sh con scripts/one_driver_v7_20261015.py.
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v7_20261015
mkdir -p $OUT/logs
for s in 01 02 03 04 05; do
  ~/.venvs/f1-drl-v2/bin/python -u scripts/one_driver_v7_20261015.py screen \
    --agents-dir trained_models/rl_agents_onedriver_v7_20261015/seed$s \
    --models-dir trained_models/rl_agents_onedriver_v7_20261015/seed$s/one_driver \
    --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
    > $OUT/logs/screen_seed$s.log 2>&1
  echo "seed$s rc=$?"
done
