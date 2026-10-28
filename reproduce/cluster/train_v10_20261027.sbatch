#!/bin/bash
# 2026-10-27 — Array de SLURM del multi-piloto v10 (Figs. 6-7). Como train_v3_20261007.sbatch, pero con los entornos en
# un solo proceso (F1_VEC_DUMMY=1 -> DummyVecEnv, mismo buffer y semillas) para respetar 4 CPU por tarea (regla del autor
# del 27-oct; con SubprocVecEnv cada tarea usa ~6 CPU aunque pida 4). Medido en el Mac (PPO Saudi, 30k pasos): 1705 fps
# (subproc 4580) y 0.53 GB. Memoria: prueba de 2 tareas con 2G; el resto con 1.2 x el pico medido
# (reproduce/cluster/RECURSOS_CLUSTER.md).
#     sbatch --job-name=f1v10t --array=<i,j> reproduce/cluster/train_v10_20261027.sbatch reproduce/cluster/tasks_v10.tsv
#     sbatch --job-name=f1v10 --mem=<1.2 x pico> --array=1-400%40 reproduce/cluster/train_v10_20261027.sbatch reproduce/cluster/tasks_v10.tsv
#SBATCH --job-name=f1v10
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=20:00:00
#SBATCH --exclude=mn021,mn023
#SBATCH --output=logs_slurm/%x_%A_%a.out

module purge >/dev/null 2>&1 || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 F1_VEC_DUMMY=1
bash reproduce/cluster/run_task_v3_20261007.sh "$1" "$SLURM_ARRAY_TASK_ID"
