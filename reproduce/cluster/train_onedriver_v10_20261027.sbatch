#!/bin/bash
# 2026-10-27 — Array de SLURM del mono-piloto v10 (Fig. 5 / Tabla N.1), = train_onedriver_20261011.sbatch con recursos
# ajustados al consumo real de f1od9r2 (sacct: MaxRSS maximo 805 MB, maximo 1 h 20 min con 2 CPU): --mem = 1.2 x pico.
# Uso, desde ~/F1DLR (primero una prueba de 2 tareas):
#     sbatch --job-name=f1od10t --array=1-2 reproduce/cluster/train_onedriver_v10_20261027.sbatch reproduce/cluster/tasks_onedriver_v10.tsv
#     sbatch --job-name=f1od10 --array=3-200%40 reproduce/cluster/train_onedriver_v10_20261027.sbatch reproduce/cluster/tasks_onedriver_v10.tsv
#SBATCH --job-name=f1od10
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=03:00:00
#SBATCH --exclude=mn021,mn023
#SBATCH --output=logs_slurm/%x_%A_%a.out

module purge >/dev/null 2>&1 || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash reproduce/cluster/run_task_v3_20261007.sh "$1" "$SLURM_ARRAY_TASK_ID"
