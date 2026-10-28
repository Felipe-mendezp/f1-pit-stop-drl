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
| multi-piloto v10, DQN | 1900M (1,2 × 1,58 GB) | 4 | ídem, job f1v10d (%20) |
| multi-piloto v10, A2C/TRPO | 550M (1,2 × 434 MB) | 4 | ídem, job f1v10o (%40) |
| post-entrenamiento v10 | 1G (1,2 × 822 MB) | 4 | `train_v10_20261027.sbatch` + `--mem=1G` (tareas `tasks_v10sm.tsv`) |

## Mediciones v10 [V]

| Trabajo | Tarea | MaxRSS | Duración |
|---|---|---|---|
| f1od10t (mono-piloto, 2 CPU) | Bahrain PPO, RPPO semilla 1 | 352 MB (sstat, en curso) | ≈ 25 min |
| f1v10t (multi-piloto, DummyVecEnv, 4 CPU) | Saudi PPO semilla 1 | 403 MB | 1 h 26 min |
| f1v10t (ídem) | Saudi RPPO semilla 1 | 557 MB | 2 h 15 min |

| f1v10t2 (ídem) | Singapore DQN / A2C / TRPO semilla 1 | 1582 / 423 / 434 MB | 1 h 28 / 1 h 18 / 1 h 35 |

| f1v10 / f1v10d / f1v10o (27-oct, 216 tareas completas antes de cancelar) | PPO+RPPO / DQN / A2C+TRPO | máx. 687 / 1606 / 436 MB | máx. 4 h 00 / 1 h 49 / 1 h 56 |

**28-oct, relanzamiento con el árbol `simulation_v10_20261028`:** f1v10b PPO/RPPO `--mem=830M` (1,2 × 687 MB; el pedido
de 700M quedaba justo), f1v10bd DQN `--mem=1950M` (1,2 × 1606 MB) y f1v10bo A2C/TRPO `--mem=530M` (1,2 × 436 MB).

Con DummyVecEnv, el multi-piloto usa ~0,5 GB por tarea en lugar de ~8 GB con SubprocVecEnv, y 1 CPU efectiva.

**28-oct, pruebas v10c** (f1v10ct, f1od10ct; mismo código y tamaño que v10d) [V]:

| Tarea | MaxRSS | Pedido | Duración |
|---|---|---|---|
| multi Singapore HUL DQN, semilla 1 | 1582 MB | 1950M | 1 h 22 |
| multi Saudi NOR RPPO, semilla 1 | 651 MB | 830M | 1 h 41 |
| mono Bahrain PPO, semilla 1 (2 CPU) | 445 MB | 500M | 46 min |
| mono Bahrain RPPO, semilla 1 (2 CPU) | 358 MB | 500M | 15 min |

**Pedidos v10d** (aprobados el 28-oct):
- multi-piloto: PPO/RPPO 830M, DQN 1950M, A2C/TRPO 530M, con 4 CPU;
- mono-piloto: **540M** (1,2 × 445 MB), con 2 CPU.

Las tareas son `tasks_v10d_20261028.tsv` y `tasks_onedriver_v10d_20261028.tsv`.

**28-oct (noche), v10eGQ** (versión adoptada; tareas `tasks_v10eGQ_20261028.tsv` y
`tasks_onedriver_v10eGQ_20261028.tsv`, no lanzadas). Mismo código y tamaño que v10d, así que se esperan los mismos
pedidos (PPO/RPPO 830M, DQN 1950M, A2C/TRPO 530M; mono 540M con 2 CPU). Prueba en curso: `18033848` (multi 52, 71,
`--mem=2G`) y `18033849` (mono 6, 7). La prueba 18033607/08 falló porque un `rsync -R --delete` borró
`~/F1DLR/trained_models/` en el cluster; se repusieron solo los árboles v10eGQ y los 16 destinos de sus enlaces
(`simulation/`, `simulation_v6_20261014/`). **No usar `--delete` hacia el cluster.**

**29-oct, v10eGQ con adelantamiento v10f.**
- **Prueba 18033848/49** (árboles anteriores al rastreo; sirve para fijar la memoria):
  - multi: PPO 0,44 GB, DQN 1,6 GB;
  - mono: 0,40 GB (RPPO y PPO);
  - el código y el tamaño son los mismos de v10d.
- **Mono-piloto lanzado:** `18037137` (`f1od10e`, 200 tareas, `tasks_onedriver_v10eGQ_20261028.tsv`).
  - Las tareas 1–8 partieron con 500M. Fue un error mío: el pico de v10d es 445 MB, así que correspondía 540M.
  - Las tareas 9–200 se pasaron a 540M con `scontrol update JobId=18037137 MinMemoryNode=540`.
  - Si alguna de las 1–8 termina en OUT_OF_MEMORY, relanzarla con 540M: no deja `.done`, así que no se salta.
- **Multi pendiente:** espera el OK del autor. Usar `tasks_v10eGQf_20261029.tsv` con la memoria por algoritmo de v10d.
  Los índices de cada grupo están en `groups_v10eGQf_20261029.txt`.

**29-oct, mono-piloto DQN:** 500–540M no alcanza (tarea 8 OOM a 498 MB; la 13 iba en 516 MB a los 5 min). El pico de
DQN mono en el arreglo anterior fue ≈ 826 MB → **1000M** (1,2 ×). Relanzadas: 18037186 (3, 8, 13, 18, 23, 28, 33) y
18037284 (`--array=38-198:5%20`). El resto del mono (PPO/RPPO/A2C/TRPO) sigue en 540M (≈ 320–400 MB).
