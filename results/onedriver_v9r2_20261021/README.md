# Mono-piloto v9r: diagnostico de las gaps grandes (A2C Hungria, local) — 2026-10-21

Problema: `../onedriver_v9r_20261021/README.md` (on-policy con parada en la vuelta 6, gaps de decenas de segundos).

Pruebas (A2C Hungria, 5 semillas, mismos argumentos que `tasks_onedriver_v9r.tsv`, ~10 min; screening con
`../onedriver_v9r_20261021/screen_v9r_20261021.py`). Gap frente a la QMIP con juegos (5924,62 s) [verificado]:

| variante | gaps s01-s05 | plan tipico |
|---|---|---|
| v8 (sin juegos, referencia) | elegida 0,000; mediana 5 semillas ~0 | C5 L23 C4 L47 C5 |
| v9r (cluster) | 48,3 48,3 48,3 48,3 42,0 | C5 L6 C3 |
| `test_a2c_hungary/`: `tyre_sets` escalada (`F1_ONE_DRIVER_SETS_SCALE=1`, `scripts/one_driver_v9r2_20261021.py`) | 48,3 48,3 45,1 28,3 39,0 | C5 L7-13 C3 |
| `test_a2c_hungary_nosetpen0/`: `F1_NO_SET_PENALTY=0` | 33,4 38,2 32,8 21,2 23,2 (PIT-SPAM: 10-24 pedidos sin juego, que no hacen nada) | 3-4 paradas reales |

- La recompensa no es el problema: en el entorno v9r el plan de la QMIP suma -5924,6 y el del agente (C5 L6 C3) -5973,0.
- Con el flag apagado el codigo actual reproduce v8 (A2C Hungria 0,000, TRPO 0,141, A2C Bahrain 0,000).
- Politica al azar en Hungria (P(parar) = 0,75 / 0,20 / 0,05): retorno v8 -15.456 / -7.626 / -7.008, v9r -61.920 /
  -16.484 / -8.486 (47,6 / 9,1 / 1,5 pedidos sin juego por carrera).
- Curvas de evaluacion (`evaluations.npz`, semillas 1 y 4): v8 pasa de -6977 a ~-5920 hacia 1,2 M pasos; v9r llega a
  -5973 hacia 1 M y no se mueve mas; con penalizacion 0 mejora lento hasta -5946/-5959 a 5 M.
- Conclusion: ni la escala ni la penalizacion explican solas el problema; la penalizacion -1000 lo agrava (con 0 las
  gaps bajan ~15-25 s). Los on-policy quedan en un optimo local temprano. Causa de fondo sin identificar.

## Diagnosticos con juegos de sobra (21-oct, noche)

Juegos de sobra = 30 nuevos por compuesto (`test_a2c_hungary_plenty/tyre_sets_plenty_hungary_20261021.csv`): entorno
identico a v8 (300 carreras al azar: recompensas y observaciones comunes iguales; QMIP 5918,16 s como v8) salvo la
observacion. Gaps A2C Hungria s01-s05 [verificado]:

| variante | gaps | lectura |
|---|---|---|
| `test_a2c_hungary_plenty/` raw: con `tyre_sets` sin escalar, tyre_life 131 | 68,7 77,8 377,9 54,7 383,1 | falla |
| `test_a2c_hungary_plenty/` scaled: `tyre_sets` escalada, tyre_life 131 | 54,0 54,0 54,0 49,6 50,6 | falla |
| `test_a2c_hungary_ab/` A: sin `tyre_sets`, tyre_life 131 | 0,27 1,91 0,61 0,00 0,26 | **aprende (como v8)** |
| `test_a2c_hungary_ab/` B: `tyre_sets` escalada, tyre_life 71 | 53,0 54,0 18,5 7,9 54,0 | falla |

Conclusion: lo que rompe el aprendizaje es la entrada `tyre_sets` (Box continua junto a entradas one-hot), no el
inventario, la penalizacion ni el tamano de tyre_life. Por que una entrada Box escalada (aqui 3,75 = 30/8 juegos; con el inventario real <= 1, prueba
`test_a2c_hungary/`, tambien falla) produce el optimo local de la vuelta 6 no esta explicado [hipotesis abierta].

Nota: la primera corrida A/B (`test_a2c_hungary_ab/_invalido_20261021/`) importaba el entorno antes de fijar las rutas
y escribio en `trained_models/rl_agents/one_driver/Hungarian/A2C/best_model.zip` (publicado v2.0.0); restaurado
desde `Codigos/respaldo/30-09-2026 trained_models/trained_models_backup_20260930.tar.gz` (cmp identico; gap 0,000 =
Tabla N.1 publicada) y borradas las carpetas creadas (`one_driver/logs`, `one_driver/tensorboard_det`).

## Diagnostico C/D con el inventario real (21-oct, noche) [verificado]

`test_a2c_hungary_cd/obs_variant_cd_20261021.py`, inventario real v9r, -1000 sin juego, A2C Hungria, QMIP 5924,62 s:

| variante | gaps s01-s05 |
|---|---|
| C: `tyre_sets` discreta (MultiDiscrete, one-hot) | 45,1 39,0 60,8 42,0 48,3 |
| D: sin `tyre_sets` | 38,8 14,9 39,0 16,8 30,8 |

Con el inventario real ni la observacion discreta ni quitarla bastan. Hay dos factores: (1) la entrada `tyre_sets`
continua rompe el aprendizaje aun con juegos de sobra (A/B); (2) el inventario limitado por si solo (D, y antes la
penalizacion 0) deja a A2C lejos del optimo. Sin resolver.

## Diagnostico E/F (21-oct, noche) [verificado]

`test_a2c_hungary_ef/obs_variant_ef_20261021.py`, A2C Hungria, 5 semillas:

| variante | gaps s01-s05 |
|---|---|
| E: inventario real, `tyre_sets` normalizada con VecNormalize sobre la marcha (como el multi-piloto) | 59,6 73,6 47,5 27,9 60,8 |
| F: juegos de sobra, `tyre_sets` discreta (= C con juegos de sobra) | 50,6 54,0 54,7 54,0 54,0 |

Ni la normalizacion del multi-piloto ni la discretizacion arreglan la entrada `tyre_sets`.

## Exploracion autonoma (21-oct, noche; el autor ausente) — `explore/`

Wrapper general `explore/obs_variant_gen_20261021.py` (OBS_MODE, SETS_CSV, HP, NORM_REWARD, REUSE) y
`explore/run_round_20261021.sh <nombre> <algo>` (entrena 5 semillas con los args de `tasks_onedriver_v9r.tsv` y hace
screening). Cada carpeta tiene `config.txt` y `summary.tsv`. A2C Hungria, 5 semillas. [verificado]

| ronda | variante | gaps s01-s05 | lectura |
|---|---|---|---|
| 1 | G0: sobra, sin `tyre_sets` (control del wrapper) | 0,27 1,91 0,61 0,00 0,26 | = A, wrapper valido |
| 1 | G1: sobra, `tyre_sets` = ceros constantes | 2,08 0,01 0,53 0,07 0,82 | aprende: no es un error mecanico |
| 1 | G2: sobra, solo el conteo de juegos que quedan | 58,2 184,9 69,0 72,8 407,7 | **el conteo rompe el aprendizaje** |
| 2 | J: G2 con ent_coef 0,2 | 62,0 151,7 248,0 398,3 264,6 | mas exploracion no ayuda |
| 2 | K1: cantidades reales de TSU con juegos nuevos, sin `tyre_sets`, -1000 | 36,9 0,27 14,9 23,7 18,5 | la edad de los blandos usados no es la causa |
| 2 | H: inventario real, sin `tyre_sets`, penalizacion 0 | 0,97 0,88 2,13 0,00 0,59 (PIT-SPAM de pedidos sin juego, que no paran) | sin -1000 el tiempo es casi optimo |

Curvas de evaluacion G1 vs G2 (semillas 1 y 3): G1 llega a -5920 hacia 1-2 M pasos; G2 sigue en -6300 a 2 M y en
-5976/-6091 a 5 M (aprende mucho mas lento).

Lectura: (1) el conteo de juegos que quedan en la observacion rompe/ralentiza el aprendizaje aun cuando no importa
(juegos de sobra) y en cualquier codificacion (Box, escalada, normalizada, one-hot); (2) con inventario limitado la
penalizacion -1000 por pedir un compuesto sin juegos es lo que aleja a A2C del optimo (H vs D). Por que el conteo
perjudica no esta explicado [hipotesis: separa estados equivalentes segun el numero de paradas previas y el agente
pierde la generalizacion entre ellos].

### Juegos reutilizables (idea del autor, 21-oct noche) [verificado]

`REUSE=1` en `explore/obs_variant_gen_20261021.py`: el juego que se saca vuelve al inventario con su edad (edad al
montarlo + vueltas corridas) y puede volver a montarse; al parar se monta el menos gastado del compuesto pedido (que puede
ser el mismo que se acaba de sacar). Con el inventario por reglamento todo piloto tiene >= 1 juego de cada compuesto en
cada GP (verificado en `data/tyre_sets_2024_reg_20261021.csv`), asi que nunca falta juego y la penalizacion -1000 nunca
se activa. Observacion `ages_s`: por compuesto, la edad del juego que se montaria si para ahora / 20 (sin conteos).
Comprobado con una carrera guiada (blando sale con 12 = 2 + 10 vueltas y vuelve al inventario; el unico medio se
vuelve a montar con su edad, sin penalizacion).

| ronda | variante (A2C, inventario real) | gaps s01-s05 | v9r (elegida / mediana) |
|---|---|---|---|
| 3 | R1 Hungria: REUSE + `ages_s` | 0,06 0,51 0,00 0,43 0,18 | 42,0 / 48,3 |
| 3 | R2 Hungria: REUSE sin `tyre_sets` | 0,17 1,32 4,09 0,97 0,40 | |
| 3 | L Hungria: sin REUSE, `avail` (1 si queda juego + edad/20) | 41,9 41,9 45,1 36,2 39,0 | |
| 4 | R1 Bahrain | 0,41 0,19 0,19 0,00 0,41 | 19,6 / 72,2 |
| 4 | R1 Belgica | 0,36 0,00 0,00 0,04 0,04 | 13,8 / 35,6 |
| 4 | R1 Holanda | 0,97 2,61 1,67 3,23 0,93 | 41,7 / 46,5 |
| 6 | R1 Singapur | 2,90 3,73 2,90 4,25 6,09 | 27,4 / 27,4 |
| 6 | R1 Estados Unidos | 0,00 0,20 0,58 0,58 0,58 | 34,8 / 45,3 |

Otros algoritmos, Hungria, R1 (v9r elegida; v8 gaps s01-s05):

| ronda | algoritmo | gaps s01-s05 | v9r elegida | v8 |
|---|---|---|---|---|
| 5 | PPO | 1,09 1,86 6,16 2,57 0,97 | 30,5 (todas 30-48) | 0,00 0,12 0,07 0,27 2,85 |
| 5 | TRPO | 18,09 2,89 1,09 47,53 60,27 | 48,3 | 2,14 13,75 4,80 8,36 0,14 |
| 5 | RecurrentPPO | 0,05 0,00 0,06 0,00 0,05 | 23,6 (todas 24-96) | 0,01 0,00 0,00 0,00 0,01 |
| 7 | DQN | 2,33 0,06 1,72 0,05 0,00 | 0,00 (0,00-12,0) | 0,00 0,92 0,00 0,00 0,00 |
| 7 | TRPO con REUSE sin `tyre_sets` (R2) | 0,97 4,09 2,89 4,45 6,80 | | |

Ronda 7, A2C Miami R1: 1,04 3,16 8,25 3,27 8,25 (v9r elegida 8,25; v8 1,44 0,13 3,21 0,24 1,86).

Matriz R1 reducida (22-oct madrugada): los pares GP x algoritmo que faltan, semillas 1-2, `explore/run_matrix_20261022.sh`
-> `explore/matrix_R1/results_20261022.tsv`.

v8 A2C (referencia, sin juegos): Singapur 0,02 2,14 0,02 3,95 1,04; Estados Unidos 0,08 0,70 0,08 0,02 0,62.

Todas las semillas R1 son sanas (1-2 paradas, cambio de compuesto). L confirma que sin reutilizacion basta la
penalizacion -1000 para romper el aprendizaje aunque el agente vea si queda juego.

Benchmark con reutilizacion: la QMIP con juegos supone que cada juego se usa una vez. `explore/dp_reuse_20261021.py`
(programacion dinamica exacta por tandas, hasta 3 paradas, mismas restricciones que el entorno) reproduce la QMIP sin
reutilizacion (Hungria 5924,62 s, Bahrain 5602,98, Belgica 5032,76, Holanda 5560,80 = QMIP del screening) y da el
mismo optimo con reutilizacion en los 8 GP del mono-piloto (`explore/dp_reuse_3stops_20261021.tsv`): reutilizar un juego gastado no conviene cuando hay juegos nuevos, asi que la
gap frente a la QMIP con juegos sigue siendo valida.

### Matriz R1 completa y TRPO R2 (22-oct madrugada) [verificado]

58 corridas mas (`explore/matrix_R1/`, semillas 1-2 en los pares que faltaban; 0 errores). Resumen por (GP, algoritmo)
en `explore/summary_R1_20261022.tsv` (`explore/summarize_R1_20261022.py`; mejor gap sana sobre las semillas corridas,
2 en la mayoria de los pares y 5 en Hungria y en A2C). Promedio de la mejor gap en los 8 GP:

| | DQN | A2C | TRPO | PPO | RecurrentPPO |
|---|---|---|---|---|---|
| R1 (juegos reutilizables + ages_s; mejor de 2-5 semillas) | 0,03 | 1,03 | 3,91 | 0,76 | 0,13 |
| v9r (cluster, mejor de 5) | 1,27 | 23,79 | 34,76 | 11,66 | 13,17 |
| v8 sin juegos (mejor de 5) | 0,00 | 0,02 | 0,50 | 0,16 | 0,00 |

Todas las corridas R1 son sanas (1-2 paradas salvo un TRPO de 4). Lo mas debil: TRPO (Bahrain 11,8, Singapur 5,3),
A2C Saudi/Singapur (~3). TRPO con REUSE sin `tyre_sets` (`explore/matrix_R2/`): Bahrain 2,88/8,88, Belgica 5,68/1,09,
Miami 13,64/4,38, Singapur 19,87/31,12 — no mejora de forma consistente; TRPO es variable (tambien en v8: Hungria
0,14-13,75). Persisten algunas paradas en la vuelta 6 (primera sin la penalizacion -15 de las vueltas 1-5) en TRPO.

Conclusion de la exploracion: con juegos reutilizables y la observacion de la edad del juego que se montaria (sin
conteos ni penalizacion por falta de juego) el aprendizaje vuelve a niveles cercanos a v8, y la QMIP con juegos sigue
siendo el optimo (DP). Implementado solo en el wrapper `explore/obs_variant_gen_20261021.py`; para adoptarlo hay que
llevarlo a `src/` (mono-piloto, multi-piloto y QMIP/inventario) y relanzar en el cluster con 5 semillas.

### Implementacion en `src/` (22-oct, decision del autor) [verificado]

Flag opt-in `F1_TYRE_REUSE=1` (con `F1_TYRE_SETS`), mono y multi-piloto:
- `src/tyre_sets.py`: `reuse_enabled`, `give_back` (el juego sacado vuelve con su edad), `has_set`, `observation_reuse`
  (edad / 20 del juego que se montaria, por compuesto; Box de 3 valores en lugar de 6).
- `f1_env_one_driver.py`: al parar devuelve el juego montado con `tyre_life + 1` y luego monta el menos gastado.
- `f1_env_all_drivers.py`: lo mismo para todos los pilotos simulados (agente y rivales; bajo SC/VSC la vuelta de entrada
  no suma, como `tyrelife`); las mascaras de compuestos con juego usan `_has_set` (el compuesto montado siempre cuenta).
- QMIP sin cambios: planifica con los juegos del inventario (incluidos los devueltos, no el montado); el DP muestra que da
  el mismo optimo.
- Envoltorio `scripts/tyre_reuse_v9r2_20261022.py`; tareas `scripts/make_tasks_v9r2_20261022.py` ->
  `reproduce/cluster/tasks_onedriver_v9r2.tsv` (200) y `tasks_v9r2.tsv` (400), salidas
  `trained_models/rl_agents_onedriver_v9r2_20261022/` y `rl_agents_v9r2_20261022/`.

Verificacion:
- Flag apagado (sin juegos, y con juegos sin reutilizacion): trayectorias identicas bit a bit a HEAD
  (`scripts/check_tyre_sets_equivalence_20261021.py`, referencia guardada con el `src/` de HEAD). La referencia del
  21-oct ya no coincidia en el multi-piloto tampoco con HEAD (cambio posterior), por eso se rehizo.
- `scripts/check_tyre_reuse_20261022.py` (`check_env`, acciones aleatorias; 4 GP multi x 4 carreras, 4 GP mono x 200):
  nunca falta juego, todo piloto conserva su numero de juegos (inventario + montado), la observacion es la esperada, y se
  vuelven a montar juegos usados.
- El screening de los 5 agentes R1 A2C Hungria con `src/` (sin el envoltorio de exploracion) da exactamente las mismas
  gaps y planes (0,057 0,506 0,000 0,429 0,176).
- Humo de entrenamiento con las tareas v9r2: mono A2C, multi PPO y DQN (observacion `tyre_sets` de 3 valores, VecNormalize).
