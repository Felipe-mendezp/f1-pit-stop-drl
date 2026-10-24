# Mono-piloto v9r (Fig. 5 / Tabla N.1), seleccion de semillas — 2026-10-21

v9r = v8 + juegos de neumaticos por reglamento (`F1_TYRE_SETS = data/tyre_sets_2024_reg_20261021.csv`,
`F1_NO_SET_PENALTY = -1000`). Agentes: `trained_models/rl_agents_onedriver_v9r_20261021/` (job `f1od9r`, 200/200 OK con
`check_tasks_20261015.sh` en el Mac; las 200 con la linea `[v9]` del inventario por reglamento).

## Orden

1. `bash results/onedriver_v9r_20261021/run_screen_20261021.sh` -> `screen_seed0X.tsv`, `degenerate_seed0X.tsv`,
   `events_seed0X.tsv` (13 s). `screen_v9r_20261021.py` = screening v8 con los juegos y `F1_QMIP_SYMMETRY=full`;
   anota la vuelta 0 y cada pedido de compuesto sin juegos (la gap es solo tiempo y no los ve).
2. `~/.venvs/f1-drl-v2/bin/python results/onedriver_v9r_20261021/select_seeds_20261021.py` -> `selection_20261021.tsv`,
   `events_summary_20261021.tsv`, `selected/` (symlinks). Regla v8 (sane = cambia de compuesto y 1-4 paradas; menor
   gap) + sin pedidos sin juegos.

## Resultados (verificados en los TSV)

- 197/200 sane (3 PIT-SPAM: Belgian A2C s3, Dutch RPPO s5, Miami TRPO s5); 0 pedidos sin juegos (vuelta 0 ni carrera),
  tampoco la QMIP; compuesto de partida = plan en 200/200; accion 0 en la vuelta 0: ninguna.
- **Las gaps empeoran mucho frente a v8** (media sobre los 8 GP de la semilla elegida / mediana de las 5 semillas):

| | A2C | DQN | PPO | RPPO | TRPO |
|---|---|---|---|---|---|
| v8 elegida | 0.02 | 0.00 | 0.16 | 0.00 | 0.50 |
| v9r elegida | 23.79 | 1.27 | 11.66 | 13.16 | 34.76 |
| v8 mediana | 0.42 | 0.10 | 0.69 | 0.00 | 4.02 |
| v9r mediana | 36.12 | 8.31 | 19.77 | 26.34 | 52.37 |

- Patron: parada en la vuelta 6 (la primera sin la penalizacion de parada temprana) y luego una tanda hasta el final,
  o varias paradas cada 2 vueltas. Modelos con parada en la vuelta 6, v9r / v8: A2C 34/40 / 0, RPPO 30/40 / 0,
  TRPO 23/40 / 1, PPO 9/40 / 0, DQN 2/40 / 0. La QMIP con juegos nunca para antes de la vuelta 15.

## Causa probable [hipotesis, no probada]

La observacion nueva `tyre_sets` (Box: juegos que quedan 0-5 y edades 0-21 vueltas) entra **sin normalizar** al
mono-piloto: `train_one_driver_paperhp_20260924.py` usa `VecNormalize(norm_obs=False)` porque antes todas las
observaciones eran discretas (one-hot) [verificado]. En el multi-piloto `tyre_sets` si se normaliza
(`NORM_OBS_KEYS` de `train_rl_agents.py`) [verificado]. Las entradas de magnitud 5-20 junto a one-hot 0/1 afectarian
sobre todo a los metodos on-policy; DQN casi no cambia. En el mono-piloto (determinista, mismo inventario en cada
episodio) `tyre_sets` no agrega informacion que no este en la historia.
