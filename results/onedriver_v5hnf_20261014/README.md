# Mono-piloto v5h SIN filtro de compuestos (2026-10-14)

Decision del autor (2026-10-14): se elimina el filtro de compuestos < 5 % (homologado con el multi-piloto; los
compuestos poco usados quedan calibrados por el GLS opcion3h). Envoltorio: `scripts/one_driver_v5hnf_20261014.py`.

- Agentes: `trained_models/rl_agents_onedriver_v5hnf_20261014/seed0{1..5}` (job f1odnf, 125 tareas = Saudi, Miami,
  Hungarian, Dutch, US x 5 algos x 5 semillas; 125/125 .done). Bahrain, Belgian y Singapore no tenian compuestos
  filtrados: se reutilizan los agentes y el screening v5h (`results/onedriver_v5h_20261013`).
- Simulador `trained_models/simulation_onedriver_opcion3h8_20261012`; filtro `data/compound_filter_one_driver_none_20261014.json`.
- `run_screen_20261014.sh` -> `screen_seed0X.tsv` (25/25 sane por semilla en los 5 GP).
- `select_seeds_20261014.py` -> `selection_20261014.tsv` y `selected/one_driver/<GP>/<algo>/best_model.zip` (symlinks a
  ambos arboles; 40/40 sane).
- `run_fig5_tableN1_20261014.sh` -> `fig5/figure_5.{png,pdf}` y `tableN1/table_M1.{csv,tex}`. No copiados al paper.
