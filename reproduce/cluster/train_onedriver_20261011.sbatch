#!/bin/bash
# 2026-10-11 — Array de SLURM del mono-piloto v5 (Fig. 5 / Tabla N.1). Una tarea = un modelo
# (GP x algoritmo x semilla). Entorno sin rivales (microsegundos por paso): menos CPU y memoria que el
# multi-piloto. Uso, desde ~/F1DLR:
#     sbatch --job-name=f1od --array=1-200%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v5.tsv
# 2026-10-12 (opcion3h en los 8 GP; Saudi va en tasks_onedriver_v5h_saudi.tsv, job f1odh):
#     sbatch --job-name=f1odh8 --array=1-175%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v5h8.tsv
# 2026-10-14 (sin filtro de compuestos; solo los 5 GP que tenian filtro):
#     sbatch --job-name=f1odnf --array=1-125%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v5hnf.tsv
#SBATCH --job-name=f1od
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
# 2026-10-15 v7 (8 GP, sin filtro, simulation_onedriver_v7_20261015):
#     sbatch --job-name=f1od7 --array=1-200%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v7.tsv
# 2026-10-19 v8 (v7 + penalizaciones del multi-piloto y accion 0 en la vuelta 0 = medio; scripts/one_driver_v8_20261019.py):
#     sbatch --job-name=f1od8 --array=1-200%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v8.tsv
# 2026-10-21 v9 (v8 + juegos de neumaticos reales 2024, F1_TYRE_SETS; scripts/make_tasks_v9_20261021.py):
#     sbatch --job-name=f1od9 --array=1-200%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v9.tsv
# 2026-10-21 (noche) v9r (inventario por reglamento; reemplaza a v9, cuyas tareas se cancelan):
#     sbatch --job-name=f1od9r --array=1-200%40 reproduce/cluster/train_onedriver_20261011.sbatch reproduce/cluster/tasks_onedriver_v9r.tsv
# (4 h hasta el 2026-10-12: 5 tareas de f1od se cortaron por tiempo, todas en mn021/mn023)
#SBATCH --exclude=mn021,mn023
# 2026-10-12: se excluyen mn021 y mn023 (lentos: ahi se cortaron por tiempo f1v5o 140/145 y f1od
#  58/60/61/101/182, y f1odh 4 no avanzo en 4 h; en otros nodos las mismas tareas tardan 1.5-2 h).
#SBATCH --output=logs_slurm/%x_%A_%a.out

module purge >/dev/null 2>&1 || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash reproduce/cluster/run_task_v3_20261007.sh "$1" "$SLURM_ARRAY_TASK_ID"
