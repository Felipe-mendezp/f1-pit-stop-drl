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

## Ablación Hungarian, Fig. 7 (30-oct)

La Fig. 7 de Hungarian casi no tiene dispersión (TSU entre 7.º y 9.º en el 90 % de las carreras, contra 6.º–13.º en
v10eGQ previo). Réplica de 300 carreras con la estrategia real (`ablacion_hungarian_20261030/`, `run_20261030.sh`) [V]:

| Variante | TSU media / de | Ancho medio del 90 %, todos |
|---|---|---|
| A: v10eGQf (vigente) | 8,11 / 0,76 | 3,3 |
| D: A con seguimiento Exp(1) | 7,58 / 0,75 | 3,7 |
| E: logit en todo gap, gap con ruido | 7,45 / 1,73 | 6,3 |
| F: logit solo ≤ 1 s, gap medio | 8,50 / 1,84 | 4,7 |
| B: logit nuevo, física previa | 7,94 / 1,84 | 6,0 |
| C: árbol previo al adelantamiento v10f | 8,08 / 2,02 | 6,4 |

La concentración viene de combinar `F1_OVERTAKE_MAX_GAP=inf` con `F1_OVERTAKE_MEAN_GAP=1`: el orden solo cambia
según el ritmo medio y las paradas, y el ruido del tiempo de vuelta queda absorbido por el bloqueo. El gap de
seguimiento 0,26 y los coeficientes nuevos del logit no lo explican.
