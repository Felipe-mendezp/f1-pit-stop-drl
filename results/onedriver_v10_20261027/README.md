# Mono-piloto v10 (Fig. 5 / Tabla N.1), 27–28-oct-2026

Simulador `trained_models/simulation_onedriver_v10_20261027` (ver `../sim_v10_20261027/README.md`). Del cambio v10 solo le
llega la Ec. 1: neumáticos y vueltas sin pasos por el pit lane, y factor 2021 de 0.740. El entorno mono-piloto no tiene SC/VSC ni rivales.

- **Agentes:** `trained_models/retrained_v10_onedriver_20261027/` (job `f1od10`).
  - 200/200 OK con `reproduce/cluster/check_tasks_20261015.sh reproduce/cluster/tasks_onedriver_v10.tsv`.
  - Los 200 logs tienen la línea `[v10 mono-piloto]` y el árbol v10.
  - Pico de memoria 389 MB; de 17 min a 54 min por tarea con 2 CPU.
- **Envoltorio:** `scripts/one_driver_v10_20261027.py` (= v8 con el árbol v10).
- **Verificación:** sin juegos, QMIP = DP en los 8 GP. Con juegos difiere, igual que en v9r2, porque el DP no modela los
  juegos.

## Orden

1. `bash results/onedriver_v10_20261027/run_screen_20261027.sh` (15 s).
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v10_20261027/select_seeds_20261027.py` → `selection_20261027.tsv`,
   `selected/`. La regla es la misma de v8, v9r y v9r2.
3. `bash results/onedriver_v10_20261027/run_fig5_tableN1_20261027.sh` → `fig5/figure_5.{pdf,png}`,
   `tableN1/table_M1.{csv,tex}`. Copiados al paper como `stints_alone.pdf` y `tab:opt_gap`.

## Resultados [V]

- 200/200 sane; ningún pedido sin juego; en todas las semillas el compuesto de partida coincide con el del plan.
- Todas las estrategias usan dos compuestos y hacen 1 o 2 paradas.

Brecha frente a la QMIP con juegos (s), media de los 8 GP:

| | DQN | A2C | TRPO | PPO | RPPO |
|---|---|---|---|---|---|
| v9r2 elegida | 0.067 | 0.912 | 3.562 | 0.809 | 0.003 |
| **v10 elegida** | **0.007** | **1.172** | **2.603** | **0.602** | **0.001** |
| v9r2 mediana de 5 semillas | 0.381 | 1.788 | 15.417 | 1.770 | 0.695 |
| **v10 mediana** | **0.385** | **1.768** | **10.718** | **1.783** | **2.108** |

- **RPPO:** óptima en 7 de 8 GP. En Saudi para una vuelta antes (0.004 s); en Belgian desplaza una vuelta entre dos
  tandas del mismo compuesto, sin efecto en el tiempo.
- **DQN:** óptima en 6 de 8 GP. En Bahrain y Belgian desplaza una vuelta, sin efecto; en Miami (0.013 s) y US (0.046 s)
  para una vuelta antes.
- **Dutch:** A2C y TRPO paran una sola vez (C1–C2, brechas 0.75 y 0.79 s).
- **Semillas elegidas:** ver `selection_20261027.tsv`.
