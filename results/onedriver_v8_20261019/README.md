# Mono-piloto v8 (Fig. 5 / Tabla N.1), 2026-10-19

v8 = v7 (`trained_models/simulation_onedriver_v7_20261015`, sin filtro, mismos hiperparametros y semillas) con la
recompensa alineada a la del multi-piloto v8 ('mix'), por decision del autor (consistencia con el texto del paper):

| Penalizacion | v7 | v8 |
|---|---|---|
| Accion 0 en la vuelta 0 | -150, compuesto sorteado | -1000, compuesto medio |
| Paradas en vueltas consecutivas | -20 | -200 |
| Parada en las vueltas 1-5 | -15 | -100 |
| Parada en la ultima vuelta | -30 | -150 |
| Sin cambio de compuesto | -1000 | -1000 (sin cambio) |

Implementacion: opciones con variables de entorno en `src/environment/f1_env_one_driver.py` (sin ellas, identico a
antes) que fija `scripts/one_driver_v8_20261019.py` (envoltorio de `one_driver_v7_20261015.py`).
Tareas: `scripts/make_tasks_onedriver_v8_20261019.py` -> `reproduce/cluster/tasks_onedriver_v8.tsv` (200), salida
`trained_models/rl_agents_onedriver_v8_20261019/`, job `f1od8` (ver `reproduce/cluster/train_onedriver_20261011.sbatch`).

Motivo adicional (verificado 19-oct): en v7 ningun agente mono-piloto (200/200) elige la accion 0 en la vuelta 0, asi
que el cambio de la vuelta 0 es de consistencia; las penalizaciones mas altas pueden cambiar algo el aprendizaje.

## Prueba local (19-oct)

- `smoke_20261019.py` / `.log`: TODO OK. (1) sin variables = codigo anterior (8 GP x 34 episodios, incluida la
  accion 0); (2) accion 0 = accion 2 (medio) salvo -1000; (3) con variables = codigo anterior con las
  penalizaciones fijadas a mano; (4) diferencias de recompensa por evento exactas; DSQ -1000 en los 8 GP.
- `smoke_train_20261019.log`: entrenamiento A2C Dutch 3000 pasos via el envoltorio, OK (imprime las 5 variables).

## Entrenamiento (19-oct)

`f1od8` 200/200 tareas OK, verificado en el Mac tras el rsync (`reproduce/cluster/check_tasks_20261015.sh
reproduce/cluster/tasks_onedriver_v8.tsv`; 400 archivos en `logs/`; las 200 con la linea `[v8 mono-piloto]` y los 5
valores de arriba; simulador `simulation_onedriver_v7_20261015` y filtro `none`).

## Orden (copias de los scripts de `results/onedriver_v7_20261015/`, preparadas 19-oct)

1. `bash results/onedriver_v8_20261019/run_screen_20261019.sh` -> `screen_seed0X.tsv`, `degenerate_seed0X.tsv`,
   `lap0_seed0X.tsv` (5 semillas en paralelo). Pasa por `screen_lap0_20261019.py`, que anota la accion de la vuelta 0
   (la accion 0 y la accion 2 dan el mismo compuesto medio; el plan del screening no las distingue).
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v8_20261019/summarize_lap0_20261019.py` -> `lap0_summary_20261019.tsv`
3. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v8_20261019/select_seeds_20261019.py` -> `selection_20261019.tsv`, `selected/`
4. `bash results/onedriver_v8_20261019/run_fig5_tableN1_20261019.sh` -> `fig5/`, `tableN1/` (no se copian al paper)

Prueba del paso 1 (19-oct, 2 modelos de seed01 con `--pairs`, salida en tmp): corre, ~9 s; la accion anotada y el
compuesto coinciden con el plan (Bahrain TRPO y Dutch A2C: accion 3 = C3).

## Resultados pasos 1-3 (19-oct, verificados en los TSV)

- Screening: 200/200 semillas sanas (5/5 en todo GP x algoritmo), 14 s. Compuesto anotado = plan en 200/200.
- Accion en la vuelta 0: 3 -> 95, 2 -> 93, 1 -> 12, **0 -> 0** (ningun modelo usa la accion 0, como en v7).
- Gap con la QMIP de la semilla elegida (s), media sobre los 8 GP, v8/v7: DQN 0.000/0.000, A2C 0.021/0.087,
  TRPO 0.497/1.049, PPO 0.160/0.088, RPPO 0.000/0.000. Mediana sobre las 5 semillas: A2C 0.420/0.215, DQN 0.101/0.003,
  PPO 0.693/0.787, RPPO 0.000/0.003, TRPO 4.019/5.790. Pares con gap elegido > 0: 17 (v7: 16).
- Paso 4 (19-oct, 17 s): `fig5/figure_5.png`, `tableN1/table_M1.csv`. Tabla N.1 = gap de la semilla elegida (igual a la
  seleccion). Fig. 5: a diferencia de v7, TRPO Bahrain termina con el compuesto duro (C1); el resto solo medio y blando.
  Gaps v7 -> v8 por GP: TRPO Hungarian 4.79 -> 0.14, Bahrain 1.60 -> 0.67, Saudi 1.20 -> 0.41, Singapore 0.33 -> 1.54,
  Dutch 0.00 -> 0.90; PPO Belgian 0.00 -> 0.60, US 0.08 -> 0.51; A2C Bahrain 0.42 -> 0.00, Saudi 0.23 -> 0.00.

## Por que casi no se usa el duro (19-oct)

`hard_compound/hard_tradeoff_20261019.py <arbol de simulacion> <json>` (aproximacion lineal b0 + b1*vida por compuesto
desde build_qmip_params y fuerza bruta de 1-3 paradas; no es el simulador completo: en Bahrain v7 da H21 S18 S18 y la
QMIP real M/S, diferencia 0.07 s). Optimo con duro: publicado = Miami, Hungarian, Dutch, Singapore; v5, v5h8, v6 y v7 =
solo Bahrain (empate con M). El factor 0.716 de v7 no baja las pendientes b1 (v6 y v7 parecidas); el cambio grande es
publicado -> v5 (offset duro-medio mayor y menor ventaja de degradacion del duro).
