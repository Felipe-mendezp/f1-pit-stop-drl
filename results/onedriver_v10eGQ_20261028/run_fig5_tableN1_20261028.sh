#!/bin/bash
# 2026-10-28: Fig. 5 y Tabla N.1 mono-piloto v10eGQ (copia de results/onedriver_v10eGQ_20261028/run_fig5_tableN1_20261027.sh con tyre_ages_v10eGQ y one_driver_v10eGQ). v10: con las semillas de selected/ (copia de results/onedriver_v9r2_20261022/run_fig5_tableN1_20261022.sh con el envoltorio one_driver_v10).
# results/onedriver_v8_20261019/run_fig5_tableN1_qmipsym_20261021.sh con el envoltorio v9r2
# (tyre_reuse_v9r2 -> tyre_sets_v9 -> one_driver_v8) y salida en esta carpeta.
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_QMIP_SYMMETRY=full F1_PAPER_FONTS=1 F1_FIG5_GRID=2x4
SEL=results/onedriver_v10eGQ_20261028/selected
OUT=results/onedriver_v10eGQ_20261028
W="scripts/tyre_ages_v10eGQ_20261028.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/one_driver_v10eGQ_20261028.py"
PY="$HOME/.venvs/f1-drl-v2/bin/python -u"
mkdir -p $OUT/logs $OUT/fig5 $OUT/tableN1
$PY $W fig5 --agents-dir $SEL --models-dir $SEL/one_driver --output $OUT/fig5/figure_5.png > $OUT/logs/fig5.log 2>&1; echo "fig5 rc=$?"
$PY $W m1 --agents-dir $SEL --out-dir $OUT/tableN1 > $OUT/logs/tableN1.log 2>&1; echo "m1 rc=$?"
