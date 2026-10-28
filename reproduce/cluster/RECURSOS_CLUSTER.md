# Recursos de los trabajos en el cluster (NLHPC)

**Regla del autor (27-oct-2026):**
- `--mem` como máximo 1,2 × el pico observado (MaxRSS de `sacct`) del mismo tipo de trabajo;
- `--cpus-per-task` como máximo 4.

**Acceso:** `ssh nlhpc`, `rsync ... nlhpc:F1DLR/`. Es un alias de `~/.ssh/config` con la llave `~/.ssh/id_ed25519_nlhpc`
y ControlMaster. Con la IP sin `-i` falla por publickey.

## Picos observados [V]

`sacct -u cthraves -S 2026-10-20 --format=JobName,State,MaxRSS,ReqMem,AllocCPUS,Elapsed -P`, consultado el 27-oct-2026.

| Trabajo | Qué es | Tareas | MaxRSS máx (MB) | p95 (MB) | Pedido | CPU | Duración máx |
|---|---|---|---|---|---|---|---|
| f1od9r2 | mono-piloto v9r2 (Fig. 5) | 200 | 805 | 735 | 8G | 2 | 1 h 20 min (media 25 min) |
| f1od9r | mono-piloto v9r | 200 | 882 | 808 | 8G | 2 | |
| f1v9r2 | multi-piloto v9r2 (Figs. 6–7) | 400 | 8016 | 7793 | 8G | 8 | |
| f1v9r | multi-piloto v9r | 400 | 8080 | 7842 | 8G | 4 | (393/400 OK) |
| f1v9r2sm | post-entrenamiento con largada sorteada (App. Q) | 48 | 822 | 708 | 1500M | 4 | ≈ 30 min |
| f1v9r2smt | prueba del anterior | 2 | 608 | 601 | 3G | 4 | |

**Mac (M1 Pro, 27-oct):** el mono-piloto PPO con 50 000 pasos (10 envs, 2 hilos) tuvo un pico de 435 MB.

## Pedidos v10

| Trabajo | `--mem` | `-c` | sbatch |
|---|---|---|---|
| mono-piloto v10 | 1G (1,2 × 805 MB) | 2 | `train_onedriver_v10_20261027.sbatch` |
| multi-piloto v10 | 10G (1,2 × 8,1 GB) | 4 | por crear |
| post-entrenamiento v10 | 1G (1,2 × 822 MB) | 4 | por crear |
