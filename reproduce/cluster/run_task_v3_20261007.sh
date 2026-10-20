#!/usr/bin/env bash
# 2026-10-07 — Ejecuta UNA tarea (línea N) de reproduce/cluster/tasks_v3_<calib>.tsv.
#     bash reproduce/cluster/run_task_v3_20261007.sh <tasks.tsv> <N>
# N empieza en 1 (= SLURM_ARRAY_TASK_ID). Se salta si ya existe <log>.done (reanudable).
set -euo pipefail
cd "$(dirname "$0")/../.."
TASKS="$1"; N="$2"
LINE="$(sed -n "${N}p" "$TASKS")"
[ -n "$LINE" ] || { echo "sin tarea $N en $TASKS"; exit 0; }
IFS=$'\t' read -r NAME LOG SIM CMD <<< "$LINE"

source reproduce/env_paper_v3_20261007.sh
export F1_SIMULATION_DIR="$SIM"
unset F1_RL_AGENTS_DIR
PY="${PYTHON:-$HOME/.venvs/f1-drl-v2/bin/python}"
[ "$PY" = python ] && PY="$HOME/.venvs/f1-drl-v2/bin/python"

if [ -f "$LOG.done" ]; then echo "$NAME ya terminada"; exit 0; fi
mkdir -p "$(dirname "$LOG")"
echo "$(date '+%F %T') $NAME en $(hostname)  SIM=$SIM  REVEAL=$F1_SC_REVEAL" | tee "$LOG"
T0=$(date +%s)
eval "\"$PY\" $CMD" >> "$LOG" 2>&1
echo "$(( $(date +%s) - T0 )) s" > "$LOG.done"
echo "$(date '+%F %T') $NAME OK ($(( ($(date +%s) - T0) / 60 )) min)"
