#!/usr/bin/env bash
# 2026-10-06 — Agentes y QMIP publicados (creen la calibracion publicada) evaluados en un "mundo" con las
# calibraciones de neumatico opcion2 / opcion3, sin ruido extra (Figs. 6 y 7). Dos carriles en paralelo.
cd "$(dirname "$0")/../.."
source reproduce/env_paper.sh >/dev/null
export PYTHON=$HOME/.venvs/f1-drl-v2/bin/python MPLBACKEND=Agg
S=scripts/evaluate_coef_uncertainty_20261005.py; L=results/coef_uncertainty_20261005
A="--tag publicado --factor 0 --no-est-noise"
( $PYTHON $S --fig 7 $A --world opcion2 > $L/fig7_mundo-opcion2.log 2>&1
  $PYTHON $S --fig 6 $A --world opcion2 > $L/fig6_mundo-opcion2.log 2>&1 ) &
( $PYTHON $S --fig 7 $A --world opcion3 > $L/fig7_mundo-opcion3.log 2>&1
  $PYTHON $S --fig 6 $A --world opcion3 > $L/fig6_mundo-opcion3.log 2>&1 ) &
wait
echo "$(date '+%F %T') WORLD DONE"
