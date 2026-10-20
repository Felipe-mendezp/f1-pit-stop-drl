# Bahrain DQN, descalificaciones en la Fig. 7 v8 — diagnostico 2026-10-19

Agente: DQN Bahrain_ALO, semilla 8 (reglas 5/7/8; validacion 1.5 % DSQ). Fig. 7 v8: 11.6 % de carreras sin cambio de
compuesto (58/500; paran 2 veces, en las mismas vueltas que las carreras validas, p. ej. 19 y 43). Fig. 6: 1.8 %.
Protocolo de `results/world2024_v7_20261015/bahrain_trpo_dsq/` con las variables v8 (`run_20261019.sh`).

Resultados [V]:
- Fig. 7 Bahrain DQN por semilla (500 carreras; pos. media, % DSQ): 1: 7.91, 0 | 2: 8.04, 0 | 3: 8.13, 6.2 |
  4: 8.34, 0 | 5: 7.60, 0.2 | 6: 7.36, 0 | 7: 8.72, 0.2 | 8: 8.60, 11.6 | 9: 8.53, 0 | 10: 9.00, 0. QMIP 7.29.
- La traza de `run_20261019.sh` (trace_s08_*) reproduce la Fig. 7 (8.60, 11.6 %) pero `trace.csv` quedo vacio: DQN
  redefine predict y `diag_20261018.py` solo envuelve BaseAlgorithm.predict. `diag_dqn_20261019.py` lo corrige (DQN
  determinista = BaseAlgorithm.predict); `run2_20261019.sh` repite trazas s08 Fig. 7/6 y s03 Fig. 7 (trace2_*; resumen por carrera en trace2_*_races.csv).

Trazas (trace2_*) [V]:
- Semilla 8 parte con C2 (medio) cuando casi todos los rivales parten con C3 (Fig. 6: 48/500 carreras, media 17.5
  rivales en C3; con otra mayoria parte siempre con C3). En la Fig. 7 Bahrain los 19 rivales parten con C3 (real 2024),
  asi que parte con C2 en 500/500.
- Partiendo con C2, la 1a parada es siempre C2 -> C2 (mismo compuesto; regla aun sin cumplir). La 2a parada va a C1
  (442/500, valida) o de nuevo a C2 (58/500 -> DSQ). Las DSQ ocurren cuando va mas atras en la 2a parada (posicion
  media 10.8 frente a 4.7); misma vuelta (~42). Sin SC/VSC en la Fig. 7.
- Fig. 6: DSQ 18.8 % de las carreras que parten con C2 y 0 % de las que parten con C3 (= 1.8 % total).
- Semilla 3 (6.2 %): parte con C3, 1a parada C3 -> C3, 2a a C2 (valida) o C3 (DSQ).

Interpretacion [H]: v8 si genera el estado "rivales concentrados en C3" (no existia en v7), pero en ~10 % de las
carreras; la politica de la semilla 8 en ese estado es fragil (parada al mismo compuesto y segunda parada dependiente
de la posicion). La Fig. 7 Bahrain es exactamente ese estado.

Seleccion con un filtro de DSQ en validacion (calculado sobre seed_summary v8, sin correr nada): con DSQ <= 1 % entre
las semillas que pasan, cambian 2 de 40 elecciones: Bahrain DQN 8 -> 1 (pos. val. 5.01 -> 5.05; Fig. 7 semilla 1:
7.91, 0 % DSQ) y Singapore RPPO 10 -> 4. Con <= 0.5 % cambian 6 (mas Miami A2C/DQN/PPO). Decision del autor.

## Decision (19-oct noche): filtro de descalificaciones en validacion

Decision del autor: una semilla pasa solo si, ademas de las reglas 5/7/8 en >= 95 % de las carreras, tiene <= 1 % de
carreras de validacion descalificadas (`--max-dsq 0.01` en `scripts/select_seeds_v5_20261010.py` y en
`reproduce/pipeline_v8_20261019.sh`). Motivo: la regla 5/7/8 mira solo paradas y vueltas, no el cambio de compuesto,
asi que una carrera descalificada puede contar como "cumple". Cambian 2 de 40: Bahrain DQN 8 -> 1, Singapore RPPO 10 -> 4
(verificado en SELECTED_SEED.txt). Seleccion anterior en `trained_models/rl_agents_v8_20261018/selected_r578_20261019/`;
CSV y figuras previas de Bahrain y Singapore en `results/world2024_v8_20261019/_pre_maxdsq_20261019/`.
Figs. 6-7 de esos dos GP y resumen: relanzados 21:59 (`logs/maxdsq_run.out`).
- **Resultado (22:22, verificado en los CSV):** Fig. 7 Bahrain DQN 8.60 / 11.6 % DSQ -> 7.91 / 0 %; Singapore RPPO
  7.68 -> 7.87 (0 % DSQ en ambos). Fig. 6 Bahrain DQN 5.59 / 1.8 % -> 5.52 / 0 %; Singapore RPPO 6.79 / 0.4 % -> 6.89 /
  3.0 % (la semilla 4 tenia 1.0 % en validacion, 2 de 200 carreras, justo en el umbral; la 10 tenia 2.5 %, 5 de 200). Resto sin cambio.
  v8 final, Fig. 7: DSQ maxima por par 2.8 % (Miami TRPO); por algoritmo A2C 0, DQN 0, PPO 0, RPPO 0.2, TRPO 0.4 %
  (v7: 6.7, 0, 0.2, 0.4, 12.2). Posicion media A2C 8.55, DQN 8.83, PPO 8.66, RPPO 8.97, TRPO 9.16, QMIP 8.88
  (v7: 9.78, 9.09, 8.63, 8.77, 10.11). Fig. 6: DSQ maxima por par 3.0 % (Singapore RPPO).
