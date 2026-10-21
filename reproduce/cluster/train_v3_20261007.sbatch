#!/bin/bash
# 2026-10-07 — Array de SLURM del reentrenamiento v3 (NLHPC Leftraru). Una tarea = un modelo
# (GP x algoritmo x semilla) de una calibración. Uso, desde ~/F1DLR:
#     sbatch --array=1-160%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v3_control.tsv
# Ejemplos 2026-10-12 (opcion3h en los 8 GP):
#     sbatch --job-name=f1v5h8  --array=1-140%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v5h8_ppo.tsv
#     sbatch --job-name=f1v5h8o --array=1-210%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v5h8_others.tsv
# 2026-10-14 (v5hd: pilotos nuevos en Belgian, Hungarian, Dutch y US; 5 algoritmos x 10 semillas):
#     sbatch --job-name=f1v5hd --array=1-200%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v5hd.tsv
# 2026-10-14 v6 (8 GP, pilotos v5hd + abandonos de rivales, F1_RIVAL_DNF; ver scripts/make_tasks_v6_20261014.py):
#     sbatch --job-name=f1v6 --array=1-400%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v6.tsv
# 2026-10-14 v6 Miami con SC/VSC = 0.50/0.50 (scripts/patch_miami_yf_v6_20261014.py); relanza las 50 tareas de Miami:
#     sbatch --job-name=f1v6m --array=1-50%20 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v6_miami.tsv
# 2026-10-15 v7 (TyreLife 2021 x 0.716 + K1; simulation_v7_20261015; scripts/make_tasks_v7_20261015.py):
#     sbatch --job-name=f1v7 --array=1-400%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v7.tsv
# 2026-10-18 v8 (v7 + compuestos de partida de rivales Dirichlet-multinomial, F1_RIVAL_START_ALPHA;
#  scripts/make_tasks_v8_20261018.py). Piloto (Bahrain TRPO, Dutch A2C, US A2C x 10 semillas) y completa:
#     sbatch --job-name=f1v8p --array=1-30%30 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v8_pilot.tsv
#     sbatch --job-name=f1v8 --array=1-400%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v8.tsv
#  Si el piloto ya corrio (18-oct), el resto (370) sin repetirlo:
#     sbatch --job-name=f1v8 --array=1-370%40 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v8_rest.tsv
# 2026-10-21 v9 (v8 + juegos de neumaticos reales 2024, F1_TYRE_SETS; scripts/make_tasks_v9_20261021.py): las 400
#  de una pasada (decision del autor) y con 4 CPU por tarea en vez de 8 (v8: media 1.1-1.6 h, max 3.0 h con 8 CPU;
#  con 4 se espera hasta ~2x, dentro de las 10 h). %80 x 4 CPU = la misma huella que v8 (%40 x 8); si la QOS
#  limita el numero de trabajos, usar %40. Memoria 8G (decision del autor; RSS medido ~6.1 GB en el Mac, OOM con 6G
#  el 07-oct): si alguna tarea cae por memoria, relanzar esos indices con --mem=10G (las terminadas se saltan):
#     sbatch -c 4 --mem=8G --job-name=f1v9 --array=1-400%80 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v9.tsv
# 2026-10-21 (noche) v9r = v9 con el inventario por reglamento (data/tyre_sets_2024_reg_20261021.csv); reemplaza a v9:
#     sbatch -c 4 --mem=8G --job-name=f1v9r --array=1-400%80 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v9r.tsv
# Prueba corta (partición debug, 30 min):
#     sbatch -p debug -t 00:25:00 -c 4 --mem=16G --array=1-2 reproduce/cluster/train_v3_20261007.sbatch reproduce/cluster/tasks_v3_control_smoke.tsv
# (debug admite como máximo 4 CPUs)
#SBATCH --job-name=f1v3
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
# (memoria: ~6.1 GB de RSS total en el Mac, 21 procesos = 10 entornos de entrenamiento + 10 de
#  evaluación + el principal; con 6G hubo OOM en debug el 2026-10-07; 16G hasta el 2026-10-12, bajado
#  a 10G por aviso de subutilización de RAM de NLHPC en PPO/RPPO)
#SBATCH --time=10:00:00
# (6 h hasta el 2026-10-12; DQN/A2C tardan ~1h45 en un nodo normal, se deja margen amplio)
#SBATCH --exclude=mn021,mn023
# 2026-10-12: se excluyen mn021 y mn023 (lentos: ahi se cortaron por tiempo f1v5o 140/145 y f1od
#  58/60/61/101/182, y f1odh 4 no avanzo en 4 h; en otros nodos las mismas tareas tardan 1.5-2 h).
#SBATCH --output=logs_slurm/%x_%A_%a.out

module purge >/dev/null 2>&1 || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash reproduce/cluster/run_task_v3_20261007.sh "$1" "$SLURM_ARRAY_TASK_ID"
