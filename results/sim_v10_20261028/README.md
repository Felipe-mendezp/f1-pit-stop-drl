# Simulador v10c (28-oct-2026)

v10c = v10 (`../sim_v10_20261027/`) re-estimado con **dos correcciones de compuesto** en los datos de FastF1. Ambas se
cotejaron con OpenF1 (`../compound_check_openf1_20261028/`); en 2024 también con Lapchive.

- **Belgica 2024:** la tanda final de ALO (14–44), STR (13–44), TSU (16–44) y RIC (22–44) figura como MEDIUM o SOFT,
  pero fue HARD (C2). Son 115 vueltas.
- **Canadá 2023:** TSU 36–70 sin compuesto. Se completa como HARD (C3), con la vida continuando el mismo juego. Son 35
  vueltas.
- 2023 y 2024 no tienen otras diferencias: 22 726 y 23 614 vueltas secas comparadas, sin contar la vuelta de PitIn, que
  OpenF1 asigna a la tanda siguiente. Para 2021–2022 no hay fuente independiente, porque OpenF1 empieza en 2023.

## Cómo se aplica

- `data_fix_20261028.py` envuelve `pandas.read_csv` y corrige `session_2023_V2.csv` y `session_2024_V2.csv` al leerlos.
  Los archivos de `data/` no se modifican.
- Los scripts de esta carpeta son copias sin cambios de los de `../sim_v10_20261027/`. La excepción es
  `common_v10_20261027.py`, que importa `data_fix_20261028`. Las salidas quedan en esta carpeta.
- Orden: `factor2021_v10_20261027.py`, luego `run_chain_20261028.sh` (neumáticos, Ec. 1, SC/VSC, rivales y
  adelantamiento). La cadena de pilotos (Ec. 2) es `../driver_coef_v10_20261028/run_20261028.py v10c`.
- Árboles: `scripts/build_simulation_v10c_20261028.py` → `trained_models/simulation_v10c_20261028` y
  `simulation_onedriver_v10c_20261028`.

## Cambios frente a v10 [V] (`compare_vs_v10_20261028.txt`)

| Componente | Cambio |
|---|---|
| Factor 2021 | Conjunto 0.7404 → 0.7418, sigue siendo **0.74**; conductual 0.854 → 0.865 (ee 0.087); p 0.078 → 0.067. Se usa el mismo archivo `tyrelife_2021_factors_v10.json` |
| Neumáticos 2021–23 (simulador y mono-piloto) | τ 0.722 → 0.727. En los 8 GP el desplazamiento cambia ≤ 0.006 s y la pendiente ≤ 0.0002 s/vuelta |
| Neumáticos "mundo 2024" | Belgian cambia mucho: C2 0.23 → 0.41 s; pendientes C3 0.120 → 0.142 y C4 0.137 → 0.174. Los otros 7 GP del paper: desplazamiento ≤ 0.016 s y pendiente ≤ 0.002 s/vuelta. Fuera del paper, hasta 0.04 (C5 de Australia, por el agrupamiento de pendientes) |
| Ec. 1 (sin pilotos) | ≤ 0.006 s |
| LogitStop | N 60 412 → 60 447; intercepto −5.50 → −5.41; SC/VSC ≤ 0.005; pendientes de vida ≤ 0.005 |
| K1, modelo SC/VSC, adelantamiento | sin cambio (≤ 1e-5) |
| Pilotos (Ec. 2, 2024) | Solo cambian los GP posteriores a Belgian: Dutch STR (agente) y ALO +0.05 s/vuelta; Singapore y US TSU +0.03 |
| Mono-piloto | QMIP = DP en los 8 GP (sin juegos); hay que reentrenarlo, porque cambian los neumáticos |

## Entrenamiento

- **Multi-piloto:** `reproduce/cluster/tasks_v10c_20261028.tsv` → `trained_models/retrained_v10c_20261028/`.
- **Mono-piloto:** `tasks_onedriver_v10c_20261028.tsv` → `trained_models/retrained_v10c_onedriver_20261028/`, con el
  envoltorio `scripts/one_driver_v10c_20261028.py`.
- **Generador de tareas:** `scripts/make_tasks_v10c_20261028.py`.
