# Mono-piloto v5h: screening y selección de semillas (2026-10-13)

- Agentes: `trained_models/rl_agents_onedriver_v5h_20261011/seed0{1..5}` (200 = 8 GP x 5 algos x 5 semillas).
- Simulador: `trained_models/simulation_onedriver_opcion3h8_20261012` (fijado por `scripts/one_driver_v5h_20261013.py`,
  que además restringe SELECTED_GP a los 8 GP). Los 25 jobs de Saudi se entrenaron con `simulation_onedriver_opcion3h_20261011`;
  el predictor de Saudi es idéntico en ambos árboles (md5 03f2192...).
- Filtro 5 %: `data/compound_filter_one_driver_5pct_wetA_20261011.json`.
- `run_screen_20261013.sh` -> `screen_seed0X.tsv` (gap en s vs QMIP, entorno determinista, 1 rollout). Resultado: sane 40/40 en las 5 semillas (~40 s en total).
- `select_seeds_20261013.py` -> `selection_20261013.tsv` (mejor semilla sane, mediana del gap) y `selected/one_driver/<GP>/<algo>/best_model.zip` (symlinks).
- `run_fig5_tableN1_20261013.sh` -> `fig5/figure_5.{png,pdf}` (8 GP, panel 9 oculto) y `tableN1/table_M1.{csv,tex}` (= best_gap de la seleccion). No copiados al paper.
