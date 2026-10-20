#!/bin/bash
# Fig. 5 y Tabla N.1 mono-piloto v8 desde las semillas seleccionadas (2026-10-19). Copia de
# results/onedriver_v7_20261015/run_fig5_tableN1_20261015.sh con el wrapper scripts/one_driver_v8_20261019.py.
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg
OUT=results/onedriver_v8_20261019
PY="$HOME/.venvs/f1-drl-v2/bin/python -u"
mkdir -p $OUT/logs
$PY scripts/one_driver_v8_20261019.py fig5 --agents-dir $OUT/selected \
    --models-dir $OUT/selected/one_driver --output $OUT/fig5/figure_5.png > $OUT/logs/fig5.log 2>&1; echo "fig5 rc=$?"
$PY scripts/one_driver_v8_20261019.py m1 --agents-dir $OUT/selected --out-dir $OUT/tableN1 > $OUT/logs/tableN1.log 2>&1; echo "m1 rc=$?"
