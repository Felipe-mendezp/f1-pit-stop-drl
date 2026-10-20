# Simulador v7 (Fase 2), 2026-10-15

v7 = v6 con la vida del neumático de 2021 × 0.716 (factor único, `data/tyrelife_2021_factors_v7.json`,
`F1_TYRELIFE_2021=v7`; estimación en `../tyrelife_2021_20261015/`) y el logit condicional (K1) para el compuesto de los
rivales. Diseño: `docs/DISENO_TYRELIFE_2021_20261015.md` (raíz del proyecto). Árboles:
`trained_models/simulation_v7_20261015/` (multi-piloto) y `trained_models/simulation_onedriver_v7_20261015/`
(mono-piloto), armados por `scripts/build_simulation_v7_20261015.py` desde `artifacts_v7/` (origen de cada archivo en
`sources.csv` del árbol). Nada publicado ni de v6 se modificó.

| Script | Qué hace | Control (con `F1_TYRELIFE_2021=legacy`) |
|---|---|---|
| `tyre_v7_20261015.py` | Ec. 1 por GP, offsets opcion3, pendientes opcion3h con τ por ML; también el "mundo 2024" (2021-24). Salida `tyre_<tag>/` | reproduce `coefs_opcion3h.csv` (dif. pendiente 4e-5 por τ = 0.743 redondeado) y `coefs_opcion3h_2021_24.csv` (exacto) |
| `laptime_v7_20261015.py` | `laptime_clear.pkl` = Ec. 1 re-estimada + neumáticos v7 + pilotos de v6; también `laptime_clear_hist.pkl` (pilotos históricos, entrada del adelantamiento) | reproduce el `laptime_clear` de v6 (dif. ≤ 6e-15) |
| `rivals_v7_20261015.py` | LogitStop (S0) re-estimado; K1 como `src/models/cond_logit_compound.CondLogitCompoundView` | S0 = publicado (3e-15); vista K1 = ConditionalLogit (2e-16) |
| `overtake_v7_20261015.py` | logit de adelantamiento con el procedimiento publicado, sin Emilia Romagna (no está en el simulador) | con los predictores publicados reproduce los pickles publicados (exacto); sin Emilia los otros GP cambian ≤ 0.003 |
| `smoke_rivals_v7_20261015.py` | entorno v6 vs v7, 40 carreras por GP, paradas y compuesto de los rivales | |
| `fig3_v7_20261015.py` | Fig. 3 con v7 en `fig3_v7/` (no copiada al paper) | |
| `qmip_plans_v7_20261015.py` | estrategia óptima de la QMIP desde la largada, v6 vs v7 | |
| `validation_2024_v7_20261015.py` | coeficientes de neumático v6 vs v7 contra la Ec. 1 de cada carrera 2024 | |

La Ec. 2 (coeficientes de piloto) no cambia: con un factor común a todos los compuestos, las regresiones por carrera no
cambian (verificado, dif. máx. 3e-12), así que v7 usa los de v6.

## Resultados
- Neumáticos (8 GP del paper): las pendientes suben donde hay datos de 2021 (Bahrain C2/C3 +0.016, Dutch C2/C3 +0.01,
  Hungarian C3 +0.005) y cambian ≤ 0.003 s/vuelta en el resto. Las diferencias entre compuestos nuevos se achican (Hungarian C5
  frente a C3: −0.46 → −0.28 s; Dutch C1 frente a C2: +0.08 → +0.25 s). τ: 0.743 → 0.721.
- Validación 2024: empate. Pendientes algo mejor (error medio absoluto 0.0311 → 0.0292, sesgo −0.002 → 0.000, 44
  celdas); diferencias entre compuestos algo peor en la versión ponderada (0.215 → 0.219). En los 8 GP: pendientes
  0.0359 → 0.0355, diferencias 0.310 → 0.313.
- QMIP desde la largada: igual en 6 de 8 GP. Dutch pasa de 1 a 2 paradas (S36-M36 → S25-S24-M23), Bahrain termina con
  medio en vez de duro.
- Rivales en el entorno (con `fast_rivals`, como el entrenamiento; agente sin paradas, sin abandonos): 1.55 → 1.59
  paradas por rival; primera parada algo antes; compuesto elegido blando 22 → 24 %. La parada usa la ruta rápida; el
  compuesto con K1 usa la ruta lenta del entorno (≈ 0.04 s por carrera, sin costo apreciable).
- Fig. 3 (US): mismo patrón cualitativo en las tres etapas (con outliers el blando mejora con la edad; sin outliers
  degrada pero es más lento nuevo; el modelo conjunto da el patrón esperado). El texto del paper sigue valiendo.
- Adelantamiento: γ0 cambia ≤ 0.06 y γ_PredGap ≤ 0.027 (Singapore −0.216 → −0.243); AUC igual (0.945 vs 0.947). La
  tabla del suplementario incluye Emilia Romagna, que v7 no re-estima (no está en el simulador).

## Fase 3 (15-oct, preparada; la lanza Charles)
- Rivales: se mantiene S0 (decisión del autor tras revisar S1-S3).
- Tareas: `scripts/make_tasks_v7_20261015.py` -> `reproduce/cluster/tasks_v7.tsv` (400, mismos comandos que v6 con
  `simulation_v7_20261015` y salida `rl_agents_v7_20261015/`) y `tasks_onedriver_v7.tsv` (200: 8 GP x 5 algoritmos x
  semillas 1-5, `scripts/one_driver_v7_20261015.py train`, sin filtro, salida `rl_agents_onedriver_v7_20261015/`).
- Comprobado en local: entrenamiento corto de PPO multi-piloto (Miami) y mono-piloto (Dutch) sin errores;
  `one_driver_v7_20261015.py verify`: QMIP = DP exacto en los 8 GP con v7.
- Árbol: los dos pickles de rivales (modelos globales) están una vez en Bahrain y enlazados desde los otros GP (53 MB
  en vez de 420 MB). Los demás links apuntan a `simulation`, `simulation_v6_20261014` y
  `simulation_yfv4_opcion3_20261007`, que ya están en el cluster.
- Evaluación preparada (15-oct, probada con agentes v6 sobre el simulador v7, solo para verificar el código):
  `F1_WORLD2024_TYRES=v7` en `scripts/evaluate_world2024_20261010.py`; `--calib v7` en
  `scripts/select_seeds_v5_20261010.py`; `reproduce/pipeline_v7_20261015.sh` (check | all, salida
  `results/world2024_v7_20261015/`); `scripts/make_manifest_v7_20261015.py`;
  `results/world2024_v7_20261015/compare_v6_v7_20261015.py`; mono-piloto en `results/onedriver_v7_20261015/`
  (screen, selección, Fig. 5 y Tabla N.1); apoyo a la decisión con
  `F1_WORLD2024_TYRES=v7 python scripts/run_ds_v6_sweep_20261015.py --version v7 --M 500 --workers 8 --figures`.
