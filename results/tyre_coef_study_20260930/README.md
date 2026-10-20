# Estudio de coeficientes de neumático (2026-09-30)

Copia del trabajo hecho en el scratchpad de la sesión. Desde el 04-oct, `pooled_20261004.py` alimenta los
simuladores alternativos `trained_models/simulation_tyre_opcion{2,3}_20261004/` (el publicado no se toca). Resumen y decisiones en `docs/PAPER_CODIGO_CORRESPONDENCIA.md` §5 (raíz del proyecto).
Los scripts se ejecutaban desde la raíz del repo con `~/.venvs/f1-drl-v2/bin/python` y rutas absolutas del
scratchpad; para re-correrlos hay que cambiar esas rutas por esta carpeta.

| Archivo / carpeta | Qué es |
|---|---|
| `regs_AC.py` | Ec. 1 (OLS 2021–23 + IQR) con filtro (A): excluye ediciones con > umbral de vueltas WET/INT. `base/` = sin filtro (reproduce `data/compounds_coefs_V3.xlsx`, dif. < 1e-12); `A/`, `A2/` = umbral 10 % (`A2` además guarda `race_params.csv`: LapNumber, LapNumber_2 y errores estándar). |
| `mc_AC.py` | Completación con (C) estricta: ≥ 2 anclas en offsets **y** pendientes (12 GP elegibles). Salidas `mc_C/` (solo C) y `mc_AC/` (A+C). |
| `mc_ACf.py` | (C) solo en offsets (18 GP elegibles). Salida `mc_ACf/` (A+C). `mc_A/`, `mc_base/`: completación original sobre A y base. |
| `mc_ACf_nomono.py`, `mc_nomono_off.py`, `mc_nomono_all.py` | Sin monotonía en pendientes / en offsets / en ninguno (`mc_nomono*/`). |
| `mc_opt3.py` | Opción 3: monotonía débil en pendientes (proyección al vecino) (`mc_opt3/`). |
| `mc_iso.py` | Opción 4: pendientes por regresión isotónica ponderada por 1/se² (`mc_iso/`). **Recomendada, pendiente de confirmar.** |
| `practice_slopes.py`, `B/` | (B) pendientes desde prácticas 2024 (descartada). |
| `race2024.py`, `race2024b.py`, `V/` | Ec. 1 solo con la carrera 2024 (vara de validación) y comparaciones (`V/comparison.csv`). |
| `compare.py` | Comparación de predictores de pendiente contra 2024. |
| `summarize_diag.py`, `cases.py` | Resumen y casos del barrido diagnóstico `results/decision_support_diag_20260930/`. |
| `mc_repo_reproduction/mc.log` | Log de la completación original reproducida con los datos del repo. |
| `pooled_20261004.py`, `pooled_20261004/` | (04-oct) Modelo conjunto GLS/BLUP. Opciones 2 (libre) y 3 (monótona, paso C4→C5 = C3→C4): `coefs_opcion{2,3}.csv`, validación contra 2024 (`validation_*.csv`), `paper_gps_table.csv`. |
| `compare_options_20261004.py`, `compare_20261004/` | (05-oct) Figs. 6–7: actual / control / op. 2 / op. 3 × PPO, RPPO, QMIP. Resúmenes totales, por GP, brecha con QMIP y boxplots. |
| `training_20261004.log`, `eval_20261004.log`, `control_20261005.log` | Logs maestros del reentrenamiento, la evaluación y el control (simulador publicado). |

Scripts asociados en `scripts/`: `build_simulation_tyrecoef_20261004.py`, `run_tyre_options_training_20261004.py`
(`--options opcion2 opcion3 control`), `run_tyre_options_eval_20261004.sh`, `run_ds_tyre_options_sweep_20261004.py`,
`run_tyre_control_chain_20261005.sh`. Resumen y resultados: `docs/PAPER_CODIGO_CORRESPONDENCIA.md` §5.1.
