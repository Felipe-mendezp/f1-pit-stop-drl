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
| multi-piloto v10, PPO/RPPO | 700M (1,2 × 557 MB) | 4 | `train_v10_20261027.sbatch` + `--mem=700M` (job f1v10, 28-oct) |
| multi-piloto v10, DQN/A2C/TRPO | por medir (prueba f1v10t2, tareas 21-23, 2G) | 4 | ídem |
| post-entrenamiento v10 | 1G (1,2 × 822 MB) | 4 | `train_v10_20261027.sbatch` + `--mem=1G` (tareas `tasks_v10sm.tsv`) |

## Mediciones v10 [V]

| Trabajo | Tarea | MaxRSS | Duración |
|---|---|---|---|
| f1od10t (mono-piloto, 2 CPU) | Bahrain PPO, RPPO semilla 1 | 352 MB (sstat, en curso) | ≈ 25 min |
| f1v10t (multi-piloto, DummyVecEnv, 4 CPU) | Saudi PPO semilla 1 | 403 MB | 1 h 26 min |
| f1v10t (ídem) | Saudi RPPO semilla 1 | 557 MB | 2 h 15 min |

Con DummyVecEnv, el multi-piloto usa ~0,5 GB por tarea en lugar de ~8 GB con SubprocVecEnv, y 1 CPU efectiva.
