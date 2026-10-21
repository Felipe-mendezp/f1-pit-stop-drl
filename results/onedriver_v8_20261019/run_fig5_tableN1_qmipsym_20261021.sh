#!/bin/bash
# 2026-10-21: Fig. 5 y Tabla N.1 mono-piloto v8 con la QMIP de simetria completa (F1_QMIP_SYMMETRY=full,
# src/optimization/qmip.py). Copia de run_fig5_tableN1_20261019.sh con otra salida; tambien corre verify (QMIP == DP).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_QMIP_SYMMETRY=full F1_PAPER_FONTS=1 F1_FIG5_GRID=2x4
SEL=results/onedriver_v8_20261019/selected
OUT=results/onedriver_v8_qmipsym_20261021
PY="$HOME/.venvs/f1-drl-v2/bin/python -u"
mkdir -p $OUT/logs $OUT/fig5 $OUT/tableN1
$PY scripts/one_driver_v8_20261019.py verify > $OUT/logs/verify.log 2>&1; echo "verify rc=$?"
$PY scripts/one_driver_v8_20261019.py fig5 --agents-dir $SEL \
    --models-dir $SEL/one_driver --output $OUT/fig5/figure_5.png > $OUT/logs/fig5.log 2>&1; echo "fig5 rc=$?"
$PY scripts/one_driver_v8_20261019.py m1 --agents-dir $SEL --out-dir $OUT/tableN1 > $OUT/logs/tableN1.log 2>&1; echo "m1 rc=$?"
