#!/usr/bin/env bash
# 2026-10-16 — Chequea las tareas multi-piloto de un tasks_*.tsv (v6/v7) tras el rsync al Mac (o en el cluster).
#     bash reproduce/cluster/check_tasks_multi_20261016.sh reproduce/cluster/tasks_v7.tsv
# Por tarea (nombre <ver>__seedXX__<GP_DRV>__<algo>): <log>.done, log sin Traceback/Error/Killed/CANCELLED y
# best_model.zip no vacío en <árbol>/seedXX/<GP_DRV>/<algo>/ (árbol = carpeta padre de logs/).
# Imprime solo las tareas con problemas y un resumen; exit 1 si hay alguna.
cd "$(dirname "$0")/../.."
TASKS="$1"; n=0; bad=0
while IFS=$'\t' read -r NAME LOG SIM CMD; do
  [ -n "$NAME" ] || continue; n=$((n+1)); prob=""
  SEED=${NAME#*__}; SEED=${SEED%%__*}
  GPD=${NAME#*__*__}; GPD=${GPD%__*}; AL=${NAME##*__}
  TREE=$(dirname "$(dirname "$LOG")")
  [ -f "$LOG" ] || prob="$prob sin_log"
  [ -f "$LOG.done" ] || prob="$prob sin_done"
  [ -f "$LOG" ] && grep -qE "Traceback|Error:|Killed|CANCELLED" "$LOG" && prob="$prob error_en_log"
  [ -s "$TREE/$SEED/$GPD/$AL/best_model.zip" ] || prob="$prob sin_best_model"
  if [ -n "$prob" ]; then bad=$((bad+1)); echo "$NAME:$prob"; fi
done < "$TASKS"
echo "$n tareas, $bad con problemas"
[ "$bad" -eq 0 ]
