# Simulador v10d (28-oct-2026)

v10d = v10c (`../sim_v10_20261028/`) más tres correcciones de compuesto en 2022. Se cotejaron con el live timing
y con Pirelli, la FIA y RaceFans (`../compound_check_livetiming_20261028/`). El autor aprobó el 28-oct.

| Carrera | Corrección | Vueltas |
|---|---|---|
| Bélgica 2022 | GAS 23–44 y TSU 31–44: SOFT (C4) → MEDIUM (C3) | 36 |
| Mónaco 2022 | LAT 3–19: HARD → INTERMEDIATE (se filtra como lluvia); LAT 20–22: vida 18–20 → 1–3 | 20 |
| (v10c) Bélgica 2024, Canadá 2023 | igual que en v10c | 150 |

## Cómo se aplica

- `data_fix_v10d_20261028.py` envuelve `pandas.read_csv` y corrige los CSV de 2022, 2023 y 2024 al leerlos. Comprueba
  con asserts lo que espera encontrar. Los CSV de `data/` no se modifican.
- Los scripts de la cadena son copias sin cambios de `../sim_v10_20261028/`. La excepción es `common_v10_20261027.py`,
  que importa `data_fix_v10d_20261028`.
- Orden de ejecución:
  1. `factor2021_v10_20261027.py` (con `F1_TYRELIFE_2021=none`);
  2. `run_chain_v10d_20261028.sh`;
  3. pilotos: `../driver_coef_v10_20261028/run_20261028.py v10d`;
  4. árboles: `scripts/build_simulation_v10d_20261028.py` → `trained_models/simulation_v10d_20261028` y
     `simulation_onedriver_v10d_20261028`.
- Comparación con v10c: `compare_vs_v10c_20261028.py` → `compare_vs_v10c_20261028.txt`.

## Cambios frente a v10c [V]

| Componente | Cambio |
|---|---|
| Factor 2021 | 0.7418 en ambos, sigue en 0.74 |
| τ 2021–23 | 0.7273 → 0.7274 |
| Neumáticos 2021–23 (simulador) | ≤ 0.013 s de desplazamiento (Bélgica C2 frente a C3); resto ≤ 0.004; pendiente ≤ 0.0003 |
| Neumáticos "mundo 2024" | ≤ 0.018 s (Bélgica); pendiente ≤ 0.0012 |
| Ec. 1 | Bélgica: dummy de año 2023 0.038; otros GP ≤ 0.004 |
| LogitStop | ≤ 0.0044 |
| K1 (logit condicional del compuesto al parar) | ≤ 0.047 (C(alt)[C4]): las paradas de GAS y TSU ahora eligen MEDIUM |
| Modelo SC/VSC | sin cambio |
| Adelantamiento | Bélgica ≤ 0.003 |
| Pilotos (Ec. 2, 2024) | media 0.0003, máximo 0.0026 s/vuelta (Saudi ZHO) |
| Fig. 4 | solo Bélgica "sin corrección": 0.341 → 0.327 |
| Mono-piloto | QMIP = DP en los 8 GP |

## Entrenamiento (no lanzado; espera aprobación del autor)

- Multi-piloto: `reproduce/cluster/tasks_v10d_20261028.tsv` (400 tareas) → `trained_models/retrained_v10d_20261028/`.
- Mono-piloto: `tasks_onedriver_v10d_20261028.tsv` (200 tareas) → `trained_models/retrained_v10d_onedriver_20261028/`.
  Usa `scripts/one_driver_v10d_20261028.py`, con `--mem=540M`.
- Generador: `scripts/make_tasks_v10d_20261028.py`.
- Árboles, tareas y scripts ya están en el cluster (28-oct).
- Las evaluaciones ya apuntan a v10d:
  - `select_seeds_v5_20261010.py`, calib `v10`;
  - `run_ds_v6_sweep_20261015.py`, versiones `v10` y `v10smhist`;
  - `make_tasks_v10sm_20261027.py`;
  - `evaluate_world2024_20261010.py`, `F1_WORLD2024_TYRES=v10d`;
  - `reproduce/pipeline_v10_20261027.sh`;
  - `results/decision_support_v10dw24belief_20261028/run_20261028.sh`.
- v10c (`simulation_v10c_20261028`, `tasks_*v10c*`) queda reemplazado. Nunca se entrenó, salvo las 4 pruebas.
