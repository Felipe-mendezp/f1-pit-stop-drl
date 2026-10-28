# Simulador v10 (27-oct-2026)

v10 = v7 con **paradas reales** en vez de `PitIn`/`PitOut` de FastF1, y con el **factor de TyreLife 2021 re-estimado**.
Las columnas de FastF1 registran pasos por el pit lane: el pelotón detrás del SC ("SAFETY CAR THROUGH THE PIT LANE") y las
penalizaciones, ver `../pitin_flag_check_20261027/`.

- Diseño y decisiones del autor: `docs/PROPUESTA_V10_20261027.md`, en la raíz del proyecto.
- Árboles: `trained_models/simulation_v10_20261027/` (multi-piloto) y `trained_models/simulation_onedriver_v10_20261027/`
  (mono-piloto), armados por `scripts/build_simulation_v10_20261027.py` desde `artifacts/`. El origen de cada archivo está
  en `sources.csv` del árbol. Nada de v7 ni de los modelos publicados se modificó.

## Definición común (`common_v10_20261027.py`)

- **Paso sin parada:** `PitIn` = 1 y la vuelta siguiente del piloto tiene el mismo compuesto y TyreLife + 1. Para `PitOut`
  vale lo mismo con la vuelta anterior. Se calcula con TyreLife sin reescalar.
- **Parada real:** `PitIn` = 1 y no es paso sin parada.
- **Conteo:** en 2021, 2022, 2023 y 2024 hay 133, 29, 64 y 48 pasos de entrada sin parada, respectivamente.

## Scripts (correr desde la raíz del repo)

| Script | Qué hace | Control [V] |
|---|---|---|
| `factor2021_v10_20261027.py` (`F1_TYRELIFE_2021=none`) | Factor único de TyreLife 2021 con la especificación principal de v7: física (Ec. 1 con combustible por año, sin pasos) + conductual (logit L3 con paradas reales, por evento bajo SC/VSC) + distancia mínima | `CONTROL=1` reproduce 0.660 / 0.786 / 0.716 |
| `tyre_v10_20261027.py` | Neumáticos opción 3h (2021-23 y "mundo 2024" 2021-24) sin pasos | sin parche y con factor v7, reproduce `tyre_v7` exacto (dif. 1e-16) |
| `laptime_v10_20261027.py` | `laptime_clear.pkl` (Ec. 1 + neumáticos v10 + pilotos de v6) y `laptime_clear_hist.pkl` | `CONTROL=1` reproduce el `laptime_clear` de v7 (dif. 0) |
| `yf_v10_20261027.py` | Modelo SC/VSC v4 sin pasos (también en Gin/Gout) → `yf_model_v10_20261027.csv` | `CONTROL=1` reproduce `data/yf_model_v4_20261007.csv` (dif. 5e-7) |
| `rivals_v10_20261027.py` | LogitStop por evento con paradas reales; K1 con paradas reales | `CONTROL=1` reproduce los pickles de v7 (dif. 0) |
| `overtake_v10_20261027.py` | Logit de adelantamiento con el factor v10 y la Ec. 1 histórica v10 (covariables de pit del rival sin cambio, ítem B.4) | procedimiento de v7 |
| `validate_rivals_20261027.py` | 40 carreras por GP con SC/VSC sorteados: paradas de los rivales en la 1.ª vuelta del evento | — |
| `test_vsc_first_lap_20261027.py` | Prueba de `F1_VSC_FIRST_LAP_ONLY` | sin flag para en las dos vueltas de VSC; con flag solo en la 1.ª |

## Factor de TyreLife 2021 [V]

| | v7 | v10 |
|---|---|---|
| Física (Ec. 1, combustible por año) | 0.660 (ee 0.072) | 0.659 (ee 0.071) |
| Conductual (logit L3) | 0.786 (ee 0.080), por vuelta con PitIn | **0.854** (ee 0.084), por evento con paradas reales |
| **Conjunto** | 0.716 (0.61–0.82) | **0.740** (0.63–0.85), p de compatibilidad 0.08 |

Archivo: `data/tyrelife_2021_factors_v10.json` (`F1_TYRELIFE_2021=v10`). La estimación conductual usa la misma
definición de parada y la misma agregación por evento que el LogitStop v10.

**Dónde se usa el factor y con qué valor quedó cada modelo v10:** el factor entra solo al estimar; los árboles guardan
los coeficientes ya estimados.

| Modelo | ¿Depende? | En v10 |
|---|---|---|
| Ec. 1 y neumáticos (2021-23 y mundo 2024) | sí | 0.740 (`tyre_v10`, `laptime_v10` con `F1_TYRELIFE_2021=v10`) |
| LogitStop (TyreLife × compuesto) | sí | 0.740 (`RM.TYRELIFE_ADJUST` verificado = 0.74) |
| K1 (compuesto de los rivales) | no (LapsLeft, alternativa) | — |
| Logit de adelantamiento (Ec. 1 histórica, TyreLife) | sí | 0.740 (re-estimado) |
| Modelo SC/VSC | no (sin TyreLife) | — |
| Ec. 2 (pilotos, por carrera) | no (factor común a todos los compuestos; verificado en v7) | pilotos de v6, como v7 |
| Compuestos de partida (Dirichlet), abandonos | no | sin cambio |

## Resultados [V]

- **Neumáticos:** τ 0.721 → 0.722. En los 8 GP el offset cambia ≤ 0.012 s y la pendiente ≤ 0.0015 s/vuelta.
- **Ec. 1:** PitOut de Saudi 18.67 → 18.81 s. El resto cambia ≤ 0.05 s, salvo el intercepto, que absorbe el cambio de
  neumáticos.
- **Modelo SC/VSC:** k_in 0.38 → 0.37 y k_out 0.62 → 0.70; el ritmo bajo SC/VSC no cambia. La pérdida de una parada bajo
  SC/VSC sube 1.0–1.6 s según el GP; por ejemplo, Saudi pasa de 13.1 a 14.6 s.
- **LogitStop:** SC 2.79 → 3.49 y VSC 1.82 → 3.00. La probabilidad predicha en la 1.ª vuelta de SC es 0.33 (v7 0.22; dato
  real de parar en el evento 0.39); en VSC es 0.22 (v7 0.10; real 0.22).
- **K1:** se eliminan 152 decisiones falsas, todas "no cambiar de compuesto": de 1894 se pasa a 1742. `not_change_compound`
  va de −0.83 a −1.51.
- **Adelantamiento:** respecto del publicado, la constante cambia ≤ 0.06 y la pendiente ≤ 0.027; AUC 0.945.
- **Validación de los rivales en el entorno:** 40 carreras por GP, SC/VSC sorteados, agente sin paradas.

| | v7 | v10 | Datos 2021-23 |
|---|---|---|---|
| Fracción que para en la 1.ª vuelta de SC | 0.23 | 0.34 | 0.39 |
| Fracción que para en la 1.ª vuelta de VSC | 0.14 | 0.24 | 0.22 |
| Paradas por rival | 1.69 | 1.82 | 1.88 (2024: 1.50) |

- **Mono-piloto:** sin juegos, QMIP = DP en los 8 GP con v10. Con juegos difiere, igual que en v9r2, porque el DP no
  modela los juegos.

## Entrenamiento

Ver `reproduce/cluster/RECURSOS_CLUSTER.md` para los recursos.
- **Mono-piloto:** `reproduce/cluster/tasks_onedriver_v10.tsv` y `train_onedriver_v10_20261027.sbatch`, envoltorio
  `scripts/one_driver_v10_20261027.py`, salida `trained_models/retrained_v10_onedriver_20261027/`.
- **Multi-piloto:** `tasks_v10.tsv` y `train_v10_20261027.sbatch`, envoltorio `scripts/v10_wrapper_20261027.py`
  (`F1_VSC_FIRST_LAP_ONLY=1`, `F1_VEC_DUMMY=1`), salida `trained_models/retrained_v10_20261027/`.
