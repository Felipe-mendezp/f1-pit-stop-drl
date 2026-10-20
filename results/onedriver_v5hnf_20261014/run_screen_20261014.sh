#!/bin/bash
# Screening mono-piloto v5h SIN filtro, 5 semillas (2026-10-14). Solo hay agentes de los 5 GP reentrenados;
# Bahrain, Belgian y Singapore salen MISSING y se toman de results/onedriver_v5h_20261013 (sin filtro ya en v5h) en la seleccion.
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v5hnf_20261014
for s in 01 02 03 04 05; do
  ~/.venvs/f1-drl-v2/bin/python -u scripts/one_driver_v5hnf_20261014.py screen \
    --agents-dir trained_models/rl_agents_onedriver_v5hnf_20261014/seed$s \
    --models-dir trained_models/rl_agents_onedriver_v5hnf_20261014/seed$s/one_driver \
    --out $OUT/screen_seed$s.tsv --degenerate-out $OUT/degenerate_seed$s.tsv \
    > $OUT/logs/screen_seed$s.log 2>&1
  echo "seed$s rc=$?"
done
