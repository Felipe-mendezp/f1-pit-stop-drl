# Mono-piloto v7 (Fig. 5 / Tabla N.1), 2026-10-15

Preparado antes de que termine el entrenamiento (job `f1od7`, 200 tareas = 8 GP x 5 algoritmos x 5 semillas,
`reproduce/cluster/tasks_onedriver_v7.tsv`). Simulador `trained_models/simulation_onedriver_v7_20261015`, sin filtro de
compuestos, envoltorio `scripts/one_driver_v7_20261015.py` (verify: QMIP = DP exacto en los 8 GP).

Orden, tras sincronizar `trained_models/rl_agents_onedriver_v7_20261015`:
1. `bash results/onedriver_v7_20261015/run_screen_20261015.sh` -> `screen_seed0X.tsv`
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v7_20261015/select_seeds_20261015.py` -> `selection_20261015.tsv`, `selected/`
3. `bash results/onedriver_v7_20261015/run_fig5_tableN1_20261015.sh` -> `fig5/`, `tableN1/` (no se copian al paper)

## Resultados (16-oct, verificados en los CSV)

- Entrenamiento: 200/200 tareas OK (`reproduce/cluster/check_tasks_20261015.sh`). Screening: las 200 semillas sanas (5/5 en todo GP x algoritmo).
- Tabla N.1 (gap con QMIP en segundos, semilla seleccionada), media sobre los 8 GP, v7 frente a v5hnf:
  DQN 0.000 / 0.000; A2C 0.087 / 0.031; TRPO 1.049 / 1.725; PPO 0.088 / 0.059; RPPO 0.000 / 0.000.
- Gaps no nulos más grandes en v7: TRPO Hungarian 4.79 y Bahrain 1.60 y Saudi 1.20; PPO y A2C Bahrain 0.54 y 0.42.
- Fig. 5: ninguna estrategia QMIP ni de agente usa el compuesto duro (solo medio y blando).
