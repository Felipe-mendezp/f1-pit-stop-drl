# Validación con la estrategia real 2024 en v10eGQf (29-oct-2026)

Es la misma validación que `results/real_strategy_check_20261027/` (ver su README), con el entorno de
`reproduce/pipeline_v10_20261027.sh`: árbol `simulation_v10eGQ_20261028`, mundo 2024 v10eGQ, edades de los juegos
v10eGQ, adelantamiento v10f (envoltorio `overtake_v10f_20261029.py`) y `gp_configs_fix_20261028.py`. Las semillas son
las de las Figs. 6–7 de `results/world2024_v10eGQf_20261029/`.

- Corridas: `run_one_20261029.sh <fig6|fig7> <GP_DRV>` (16 corridas, ~15 min con 2 procesos).
- Comparación: `compare_20261029.py` → `compare_{agent,all,summary}_20261029.csv`.

Resultados [V]:

| | fig7 | fig6 |
|---|---|---|
| Piloto del agente dentro del intervalo del 90 % | 7/8 (Dutch STR fuera: real 12.º, [7, 11]) | 7/8 (Miami OCO fuera) |
| Cobertura del 90 %, todos los pilotos | 80,3 % | 92,1 % |
| Spearman medio | 0,91 | 0,88 |
| Error absoluto de la mediana | 1,66 | 1,90 |

- Optimismo del replay para el piloto del agente en fig7: media simulada 8,88 frente a 10,25 real (1,37 posiciones;
  en v9r2 eran 0,85).
- Los intervalos son más angostos que en v9r2 (ancho medio del 90 % en fig7: 3,7 frente a 6,4).

La caja "Actual" de la Fig. 7 se genera así:

    F1_REALSTRAT_DIR=$PWD/results/real_strategy_check_v10eGQf_20261029 \
    F1_V9R2_DIR=$PWD/results/world2024_v10eGQf_20261029 \
        python scripts/gp_configs_fix_20261028.py results/world2024_v9r2_20261023/fig7_actual_20261027.py

Propuesta para el paper: `docs/PROPUESTA_FIGS67_V10F_20261029.md` (raíz del proyecto).
