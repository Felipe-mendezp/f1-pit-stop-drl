# Mono-piloto v10eGQ (Fig. 5 / Tabla N.1), 28-oct-2026

Copia del flujo de `../onedriver_v10_20261027/` con el simulador `trained_models/simulation_onedriver_v10eGQ_20261028`
(edad efectiva GQ, factor 2021 0,78; el adelantamiento v10f no afecta al mono-piloto, que no tiene rivales) y las edades
de los juegos 2024 en vueltas de empuje (`data/tyre_sets_2024_reg_v10eGQ_20261028.csv`, `tyre_ages_v10eGQ_20261028.py`).

- **Agentes:** `trained_models/retrained_v10eGQ_onedriver_20261028/`, bajados del cluster el 28-oct
  (trabajos 18037137 y, para DQN, 18037186 y 18037284). 200/200 `.done`, 200 `best_model.zip`; los 200 logs tienen la
  línea `[v10eGQ]` y `SIM=trained_models/simulation_onedriver_v10eGQ_20261028`. El árbol es idéntico Mac ↔ cluster (md5).

## Orden
1. `bash results/onedriver_v10eGQ_20261028/run_screen_20261028.sh` (15 s).
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v10eGQ_20261028/select_seeds_20261028.py` → `selection_20261028.tsv`, `selected/`.
3. `bash results/onedriver_v10eGQ_20261028/run_fig5_tableN1_20261028.sh` → `fig5/figure_5.{pdf,png}`, `tableN1/table_M1.{csv,tex}`.
   Copiados al paper como `stints_alone.pdf` (anterior en `stints_alone_old_v10_20261028.pdf`) y `tab:opt_gap`.

## Resultados [V]
- 200/200 sane; ningún pedido sin juego; compuesto de partida = el del plan en todas.
- En las 200 semillas: 108 con 1 parada y 2 compuestos, 87 con 2 paradas y 2 compuestos, 4 con 2 paradas y 3 compuestos,
  1 con 3 paradas (Belgian TRPO s01). Las 40 elegidas usan 2 compuestos y 1–2 paradas.

Brecha frente a la QMIP con juegos (s), media de los 8 GP:

| | DQN | A2C | TRPO | PPO | RPPO |
|---|---|---|---|---|---|
| v10 elegida | 0.007 | 1.172 | 2.603 | 0.602 | 0.001 |
| **v10eGQ elegida** | **0.072** | **1.377** | **3.587** | **0.461** | **0.016** |
| v10 mediana de 5 semillas | 0.385 | 1.768 | 10.718 | 1.783 | 2.108 |
| **v10eGQ mediana** | **0.596** | **2.214** | **6.893** | **2.503** | **0.903** |

- **RPPO:** óptima en 7 de 8; en Miami para 3 vueltas antes (0.128 s).
- **DQN:** óptima en 5 de 8 (en Bahrain desplaza una vuelta entre dos tandas del mismo compuesto). Miami para 4 vueltas
  después (0.421), US 1 vuelta después en la 1.ª parada (0.102), Belgian C3 en vez de C2 en la última tanda (0.057).
- **Dutch:** A2C, PPO y TRPO paran una vez (C1–C2) en vez de dos; brechas 1.231, 1.231 y 1.408 s.
- Ojo: la Fig. 5 reordena las tandas de los agentes para alinearlas con la QMIP (`best_reordering` en
  `scripts/plot_fig5_stints_one_driver.py`), por eso el `plan` de `screen_seed*.tsv` puede mostrar otro orden.
