# Mono-piloto v9r2 (Fig. 5 / Tabla N.1), seleccion de semillas — 2026-10-22

v9r2 = v9r + juegos reutilizables (`F1_TYRE_REUSE=1`, `src/tyre_sets.py`; diseno y exploracion en
`../onedriver_v9r2_20261021/README.md`). Agentes: `trained_models/rl_agents_onedriver_v9r2_20261022/` (job `f1od9r2`;
200/200 OK con `reproduce/cluster/check_tasks_20261015.sh reproduce/cluster/tasks_onedriver_v9r2.tsv` en el Mac;
los 200 logs con la linea `[v9r2] F1_TYRE_REUSE = 1`).

## Orden

1. `bash results/onedriver_v9r2_20261022/run_screen_20261022.sh` (17 s) -> `screen_seed0X.tsv`, `degenerate_seed0X.tsv`,
   `events_seed0X.tsv`. `screen_v9r2_20261022.py` = screening v9r con `F1_TYRE_REUSE=1`; el control de pedidos sin
   juego usa `tyre_sets.has_set` con el compuesto montado (como el entorno).
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v9r2_20261022/select_seeds_20261022.py` -> `selection_20261022.tsv`,
   `events_summary_20261022.tsv`, `selected/` (symlinks). Regla v8/v9r sin cambios.

## Resultados (verificados en los TSV)

- 200/200 sane; 0 pedidos sin juego (vuelta 0 ni carrera); compuesto de partida = plan en 200/200.
- Gap frente a la QMIP con juegos (s), media sobre los 8 GP:

| | A2C | DQN | PPO | RPPO | TRPO |
|---|---|---|---|---|---|
| v8 elegida | 0.02 | 0.00 | 0.16 | 0.00 | 0.50 |
| v9r elegida | 23.79 | 1.27 | 11.66 | 13.17 | 34.76 |
| **v9r2 elegida** | **0.91** | **0.07** | **0.81** | **0.00** | **3.56** |
| v8 mediana 5 semillas | 0.42 | 0.10 | 0.69 | 0.00 | 4.02 |
| v9r mediana | 36.12 | 8.31 | 19.77 | 26.34 | 52.37 |
| **v9r2 mediana** | **1.79** | **0.38** | **1.77** | **0.70** | **15.42** |

- Parada en la vuelta 6 (patron de v9r): v9r2 4/200 (TRPO 3, RPPO 1); v9r 98/200; v8 1/200.
- Peor elegida: Belgian TRPO 10.34 (v8 0.13). TRPO elegida > 2 s en Bahrain 4.14, Belgian, Miami 3.13, Saudi 2.79,
  Singapore 5.27; otras > 2 s: Saudi A2C 2.46 y PPO 3.22, Singapore A2C 2.90.
- Accion en la vuelta 0: 2 = 123, 1 = 58, 3 = 19 (sin accion 0).

## Fig. 5 y Tabla N.1 (22-oct)

3. `bash results/onedriver_v9r2_20261022/run_fig5_tableN1_20261022.sh` (17 s) -> `fig5/figure_5.{png,pdf}`,
   `tableN1/table_M1.{csv,tex}` (no copiados al paper). Envoltorio v9r2 (log: `[v9r2] F1_TYRE_REUSE = 1`),
   `F1_QMIP_SYMMETRY=full`, grilla 2x4 como la Fig. 5 v8 final. La Tabla N.1 coincide con `best_gap` de
   `selection_20261022.tsv` (medias DQN 0.067, A2C 0.912, TRPO 3.562, PPO 0.809, RPPO 0.003).
- QMIP con juegos: Bahrain C2-C1-C1, Saudi C4-C3, Miami C4-C3, Hungary C5-C4-C3, Belgian C4-C4-C3, Dutch C3-C3-C2,
  Singapore C5-C4, US C3-C2. RPPO y DQN reproducen el plan en 7/8 (DQN Miami 0.54 s, RPPO Miami 0.03 s).
- Desvios visibles: Dutch A2C/PPO/TRPO 1 parada C1-C2 (0.93 s); Singapore TRPO y A2C parten con C3; Saudi A2C/TRPO
  tanda final corta (C3 de 6-7 vueltas); Miami TRPO parte con duro; Belgian TRPO C2-C2-C3.
