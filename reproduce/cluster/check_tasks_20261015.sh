#!/usr/bin/env bash
# 2026-10-15 — Chequea que todas las tareas de un tasks_*.tsv terminaron bien (cluster o Mac, tras rsync).
#     bash reproduce/cluster/check_tasks_20261015.sh reproduce/cluster/tasks_onedriver_v7.tsv
# Por tarea: <log>.done, "ALL TRAINING COMPLETED" en el log, sin Traceback/Error, y best_model.zip en
# <agents-dir>/<sub>/<GP>/<algo>/ y evaluations.npz en <agents-dir>/<sub>/logs/<GP>_<algo>/ (sub = one_driver).
# Imprime solo las tareas con problemas y un resumen; exit 1 si hay alguna.
cd "$(dirname "$0")/../.."
TASKS="$1"; n=0; bad=0
while IFS=$'\t' read -r NAME LOG SIM CMD; do
  [ -n "$NAME" ] || continue; n=$((n+1)); prob=""
  [ -f "$LOG" ] || prob="$prob sin_log"
  [ -f "$LOG.done" ] || prob="$prob sin_done"
  if [ -f "$LOG" ]; then
    grep -q "ALL TRAINING COMPLETED" "$LOG" || prob="$prob log_incompleto"
    grep -qE "Traceback|Error:|Killed|CANCELLED" "$LOG" && prob="$prob error_en_log"
  fi
  AD=$(sed -E "s/.*--agents-dir ([^ ]+).*/\1/" <<< "$CMD")
  GP=$(sed -E "s/.*--gps '([^']+) Grand Prix'.*/\1/" <<< "$CMD")
  AL=$(sed -E "s/.*--algos ([^ ]+).*/\1/" <<< "$CMD")
  D=$(ls -d "$AD"/one_driver/"$GP"/"$AL" 2>/dev/null | head -1)
  if [ -z "$D" ]; then prob="$prob sin_dir_modelo"
  else
    [ -s "$D/best_model.zip" ] || prob="$prob sin_best_model"
    [ -s "$(dirname "$(dirname "$D")")/logs/${GP}_${AL}/evaluations.npz" ] || prob="$prob sin_evaluations"
  fi
  if [ -n "$prob" ]; then bad=$((bad+1)); echo "$NAME:$prob"; fi
done < "$TASKS"
echo "$n tareas, $bad con problemas"
[ "$bad" -eq 0 ]
