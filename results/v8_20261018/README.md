# v8 (2026-10-18): compuestos de partida de los rivales Dirichlet-multinomial + accion 0 en la vuelta 0

**Motivo.** En v7 los rivales sin estrategia real parten con un compuesto sorteado uniforme e independiente
(~6 por compuesto). En las carreras reales casi todos parten con el mismo; en Bahrain 2024 los 19 con C3, fuera de
distribucion para TRPO (DSQ en la Fig. 7; ver `results/world2024_v7_20261015/bahrain_trpo_dsq/README.md`).

**Cambio.** Por episodio, p ~ Dirichlet(alpha, alpha, alpha) sobre los 3 compuestos disponibles (nominacion 2024 del
simulador) y cada rival parte con p. alpha simetrico = 0.4368, MV sobre la vuelta 1 de 19 carreras en seco 2021-23 de los
8 GP agrupados (`scripts/fit_rival_start_alpha_20261018.py` -> `data/rival_start_alpha_2021_2023_20261018.json`).
Misma marginal que v7 (1/3 por compuesto). Se activa con `F1_RIVAL_START_ALPHA` (factory de
`src/environment/f1_env_all_drivers.py`); `scripts/rival_start_v8_20261018.py` la fija y encadena
`scripts/rival_dnf_v6_20261014.py` (abandonos + pilotos v5hd, igual que v7). Simulador: `simulation_v7_20261015`.
Afecta entrenamiento, seleccion de semillas y Fig. 6; la Fig. 7 sigue con los compuestos reales de 2024.

**Prueba local** (`smoke_rival_start_20261018.py`, log `smoke_rival_start_20261018.log`, N = 2000 resets por GP):
1. Sin alpha, compuestos y estado del RNG identicos al codigo anterior (8 GP x 50 semillas): OK.
2. Con alpha: marginal ~1/3; proporcion media del compuesto mayoritario 0.734 (real 2021-23: 0.756; uniforme: 0.450);
   P(>=15 de 19 iguales) 0.443 (real 0.474; uniforme 0.000); P(19 iguales) 0.097 (real 0 de 19 carreras).
3. Con compuestos reales en todos los rivales (modo Fig. 7), alpha no cambia nada (compuestos y RNG): OK.
4. Entrenamiento de humo TRPO Bahrain 3000 pasos via el wrapper: OK (`smoke_train_20261018.log`);
   alpha y abandonos llegan a los entornos en subprocesos (SubprocVecEnv spawn): OK.

## Accion 0 en la vuelta 0 (2026-10-18, tarde; decision del autor)

**Motivo.** En v7 la accion 0 en la vuelta 0 es invalida: penalizacion -600 y compuesto de partida sorteado. Varios
modelos la eligen igual al evaluar (A2C 10/10 semillas en Bahrain, Dutch, Hungarian y US; PPO 6/10 en Bahrain; TRPO a
veces), asi que su partida es aleatoria; en la Fig. 7 eso causa DSQ de A2C Dutch/US (20 % y 23.5 %) al partir con el
compuesto al que luego paran (`results/world2024_v7_20261015/a2c_dqn_diag_20261018/README.md`). SB3 no permite
enmascarar la accion en estos algoritmos.

**Cambio.** `F1_LAP0_ACTION0_MEDIUM=1`: la accion 0 en la vuelta 0 da el compuesto medio de los 3 disponibles (sin
sorteo), con penalizacion `F1_LAP0_PENALTY` = -1000 (igual a la de DSQ de `env_paper.sh`; la DSQ ademas quita el bono
final). Acciones 1-3 sin cambio; la accion 0 queda como la accion 2 menos 1000. Ambas variables las fija
`scripts/rival_start_v8_20261018.py` (si no vienen dadas). Codigo: `_handle_lap_0` y la factory de
`src/environment/f1_env_all_drivers.py`.

**Prueba local** (`smoke_lap0_20261018.py`, log `smoke_lap0_20261018.log`, 8 GP x 5 semillas, episodios completos): OK.
1. Sin la opcion, episodios identicos al codigo anterior (acciones 0-3; recompensas, observaciones, estado del RNG).
2. Con la opcion, accion 0 = medio y -1000; episodio identico al de la accion 2 salvo la recompensa de la vuelta 0.
3. Con la opcion, acciones 1-3 sin cambio. 4. Partir con el medio sin cambiar de compuesto sigue dando DSQ.
5. La factory lee las variables; llegan a los entornos en subprocesos (SubprocVecEnv spawn).
6. Entrenamiento de humo A2C Dutch 3000 pasos via el wrapper: OK (`smoke_train_lap0_20261018.log`).

**Manuscrito.** `sn-article.tex` L719 dice "Not selecting a starting compound incurs a penalty of 600": cambia si se
usa v8 (item 37 del plan; el 38 describe los compuestos de partida de los rivales).

**Tareas.** `reproduce/cluster/tasks_v8.tsv` (400, = v7 con el wrapper v8, salida `trained_models/rl_agents_v8_20261018/`)
y `tasks_v8_pilot.tsv` (30: Bahrain TRPO, Dutch A2C, US A2C x 10 semillas; lanzado solo el 18-oct, job `f1v8p`). `tasks_v8_rest.tsv` = las otras 370 (job `f1v8`, lanzado el 19-oct; al
19-oct 383 iniciadas, 343 terminadas, 0 errores). Comandos sbatch en
`reproduce/cluster/train_v3_20261007.sbatch`.

**Mono-piloto v8** (19-oct): misma recompensa que el multi-piloto, en `results/onedriver_v8_20261019/README.md`.

**Pendiente.** `reproduce/pipeline_v8_*.sh` (seleccion, Figs. 6-7) con el wrapper v8 y el arbol v8.

## Validacion y seleccion de semillas (19-oct, verificado en los CSV)

- `scripts/select_seeds_v5_20261010.py --calib v8` (nuevo; via `scripts/rival_start_v8_20261018.py`, simulador v7,
  200 carreras, semilla 2026, reglas 5/7/8 + `--ok-neutral`). Los CSV nuevos traen la columna `a0` (accion en la
  vuelta 0). Salida `results/seed_selection_v5_20261010/v8/`, elegidas en `trained_models/rl_agents_v8_20261018/selected/`,
  logs `logs/select_eval.log`, `logs/select_pick.log`. 400/400 modelos evaluados.
- Seleccion: 2 pares sin semilla que pase (Miami TRPO 0.935, Hungarian DQN 0.910); en v7 tambien 2. Semillas que pasan:
  274/400 (v7: 263). Posicion media de la elegida en validacion, media por algoritmo v8/v7: A2C 7.32/7.88,
  DQN 7.29/7.46, PPO 6.79/6.95, RPPO 6.80/6.86, TRPO 7.74/7.83 (validacion v8 vs v7: entornos distintos, no comparable 1 a 1).
- Accion 0 en la vuelta 0 (`lap0_actions_multi_20261019.tsv`): 90/400 modelos la usan en alguna carrera, 57 en todas.
  A2C la usa con las 10 semillas en Bahrain, Saudi, Hungarian, Dutch y US (Singapore 4/10); TRPO en 1-7 semillas de 7 GP;
  DQN, PPO y RPPO nunca. Entre las elegidas: 6 A2C (a0 = 0 en el 100 % de las carreras) y Dutch TRPO (100 %),
  ademas Belgian TRPO 18 %, Singapore TRPO 22 %, Hungarian TRPO 1 %. Como en v8 la accion 0 da el compuesto medio
  (mismo episodio que la accion 2 salvo -1000, smoke_lap0), no cambia posiciones ni paradas; solo la recompensa.
  [H] A2C fija la accion de la vuelta 0 temprano (politica casi determinista) y la penalizacion de un solo paso no la corrige.

## Figs. 6-7 v8 (19-oct)

`reproduce/pipeline_v8_20261019.sh` (copia de v7 con agentes v8 y las variables v8 exportadas; seleccion ya hecha)
lanzado 20:28 como `fig6 && fig7 && summary`; salida `results/world2024_v8_20261019/`, estado en `logs/status.txt`.
- 19-oct 21:07: pipeline terminado (16/16 GP x figura exit 0, resumen en `results/world2024_v8_20261019/logs/summary.log`).
  Comparacion con v7 (500 carreras por GP x algoritmo, CSV de fig6_2024_realdnf y fig7_2024):
  Fig. 7: DSQ Bahrain TRPO 96.8 % -> 0, Dutch A2C 24.8 % -> 0, US A2C 27.8 % -> 0; nueva: Bahrain DQN 0 -> 11.6 %.
  DSQ por algoritmo v7 -> v8: A2C 6.7 -> 0.0, TRPO 12.2 -> 0.4, DQN 0.0 -> 1.5, PPO 0.2 -> 0.0, RPPO 0.4 -> 0.2.
  Posicion media Fig. 7 v7 -> v8: A2C 9.78 -> 8.55, TRPO 10.11 -> 9.16, DQN 9.09 -> 8.92, PPO 8.63 -> 8.66,
  RPPO 8.77 -> 8.94, QMIP 8.88. Fig. 6: diferencias de 0.1-0.3 posiciones; DSQ <= 2.4 % en todo par (v7 <= 5.6 %).
- Bahrain DQN Fig. 7 (11.6 % DSQ): diagnostico en `results/world2024_v8_20261019/bahrain_dqn_dsq/` (en curso). Por
  semilla: solo la 8 (elegida, 11.6 %) y la 3 (6.2 %) tienen DSQ apreciable; las otras 8 entre 0 y 0.2 %.
- 19-oct noche: seleccion v8 con `--max-dsq 0.01` (decision del autor; ver `results/world2024_v8_20261019/bahrain_dqn_dsq/README.md`):
  Bahrain DQN 8 -> 1, Singapore RPPO 10 -> 4; resumen `seed_summary_..._r578_dsq1.csv`. Figs. 6-7 de esos GP relanzadas.
- **Resultado (22:22, verificado en los CSV):** Fig. 7 Bahrain DQN 8.60 / 11.6 % DSQ -> 7.91 / 0 %; Singapore RPPO
  7.68 -> 7.87 (0 % DSQ en ambos). Fig. 6 Bahrain DQN 5.59 / 1.8 % -> 5.52 / 0 %; Singapore RPPO 6.79 / 0.4 % -> 6.89 /
  3.0 % (la semilla 4 tenia 1.0 % en validacion, 2 de 200 carreras, justo en el umbral; la 10 tenia 2.5 %, 5 de 200). Resto sin cambio.
  v8 final, Fig. 7: DSQ maxima por par 2.8 % (Miami TRPO); por algoritmo A2C 0, DQN 0, PPO 0, RPPO 0.2, TRPO 0.4 %
  (v7: 6.7, 0, 0.2, 0.4, 12.2). Posicion media A2C 8.55, DQN 8.83, PPO 8.66, RPPO 8.97, TRPO 9.16, QMIP 8.88
  (v7: 9.78, 9.09, 8.63, 8.77, 10.11). Fig. 6: DSQ maxima por par 3.0 % (Singapore RPPO).
