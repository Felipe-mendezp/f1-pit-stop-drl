#!/usr/bin/env bash
# 2026-10-05: entrenamiento control (16) + evaluacion Figs 6-7 + comparacion
cd "$(dirname "$0")/.."
export PYTHON=$HOME/.venvs/f1-drl-v2/bin/python MPLBACKEND=Agg
L=results/tyre_coef_study_20260930/control_20261005.log
echo "$(date '+%F %T') START training control" >> $L
$PYTHON scripts/run_tyre_options_training_20261004.py --options control --jobs 3 >> $L 2>&1
echo "$(date '+%F %T') training finished; eval" >> $L
bash scripts/run_tyre_options_eval_20261004.sh control >> $L 2>&1
echo "$(date '+%F %T') FINAL DONE" >> $L
