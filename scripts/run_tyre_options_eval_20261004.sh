#!/usr/bin/env bash
# 2026-10-04 — Evaluacion de las opciones de coeficientes de neumatico (despues del entrenamiento de
# scripts/run_tyre_options_training_20261004.py). Por cada opcion, con su simulador y sus agentes:
#   1. Fig. 7 (replay 2024)          evaluate_all_gps_with_qmip.py  -> cache en el arbol de agentes
#   2. Fig. 6 (carreras estocasticas) evaluate_qmip_vs_drl.py       -> cache en el arbol de agentes
#   3. Apoyo a la decision: barrido + figuras por GP (run_ds_tyre_options_sweep_20261004.py)
# y al final la comparacion con la calibracion actual (compare_options_20261004.py).
# Nada escribe en trained_models/rl_agents* ni en trained_models/simulation.
#   bash scripts/run_tyre_options_eval_20261004.sh [opcion2 opcion3]
set -uo pipefail
cd "$(dirname "$0")/.."
PY_ENV="${PYTHON:-$HOME/.venvs/f1-drl-v2/bin/python}"
source reproduce/env_paper.sh   # sets PYTHON=python if unset
export PYTHON="$PY_ENV" MPLBACKEND=Agg
"$PYTHON" -c "import gurobipy, sb3_contrib" || { echo "bad PYTHON=$PYTHON"; exit 1; }
GPS="Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_SAI Belgian_NOR Dutch_RUS Singapore_HUL United_States_PIA"
OPTS="${*:-opcion2 opcion3}"
LOG=results/tyre_coef_study_20260930/eval_20261004.log
for opt in $OPTS; do
  if [ "$opt" = control ]; then export F1_SIMULATION_DIR="$PWD/trained_models/simulation"
  else export F1_SIMULATION_DIR="$PWD/trained_models/simulation_tyre_${opt}_20261004"; fi
  export F1_RL_AGENTS_DIR="$PWD/trained_models/rl_agents_tyre_${opt}_20261004"
  echo "$(date '+%F %T') === $opt: Fig. 7" | tee -a "$LOG"
  $PYTHON scripts/evaluate_all_gps_with_qmip.py --gps $GPS --force > "$F1_RL_AGENTS_DIR/logs/eval_fig7.log" 2>&1; rc=$?
  echo "$(date '+%F %T') rc=$rc ; Fig. 6" | tee -a "$LOG"
  $PYTHON scripts/evaluate_qmip_vs_drl.py --gps $GPS --force > "$F1_RL_AGENTS_DIR/logs/eval_fig6.log" 2>&1; rc=$?
  echo "$(date '+%F %T') rc=$rc ; decision support" | tee -a "$LOG"
  [ "$opt" = control ] && { echo "control: sin barrido DS" | tee -a "$LOG"; continue; }
  $PYTHON scripts/run_ds_tyre_options_sweep_20261004.py --option "$opt" --workers 5 --figures \
    > "$F1_RL_AGENTS_DIR/logs/ds_sweep.log" 2>&1; rc=$?
  echo "$(date '+%F %T') rc=$rc ; $opt done" | tee -a "$LOG"
done
unset F1_SIMULATION_DIR F1_RL_AGENTS_DIR
for opt in $OPTS; do [ "$opt" = control ] && continue; $PYTHON results/tyre_coef_study_20260930/summarize_diag.py "results/decision_support_tyre_${opt}_20261004" >> "$LOG" 2>&1; done
$PYTHON results/tyre_coef_study_20260930/compare_options_20261004.py >> "$LOG" 2>&1
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
