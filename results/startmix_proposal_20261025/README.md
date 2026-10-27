# Diagnóstico de largada forzada, agentes v9r2 — 2026-10-25

`scripts/start_compound_probe_20261025.py` sobre `trained_models/rl_agents_v9r2_20261022/selected` (PPO y RPPO, 8 GP),
200 carreras por modo, argmax, grilla real 2024, rivales y SC/VSC simulados, sin máscara de legalidad, misma semilla por
carrera en todos los modos. Envoltorio v9r2 + `env_paper_v3`, `F1_SIMULATION_DIR=trained_models/simulation_v7_20261015`.
Salida `start_probe_v9r2_20261025.csv` (log `.log`).

Resultado [V]: los 16 agentes largan siempre con el mismo compuesto; 14 de 16 quedan descalificados en el 87–100 % de las
carreras con al menos un compuesto de largada forzado (el compuesto al que paran en su propia estrategia). Excepciones:
Dutch PPO (máx. 26,5 %) y US PPO (0 %). Propuesta de post-entrenamiento: `docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md`
(raíz del proyecto).

25-oct: post-entrenamiento aprobado e implementado (`F1_START_MIX`, `scripts/finetune_startmix_20261025.py`,
`reproduce/cluster/tasks_v9r2sm.tsv`, 48 tareas); prueba de 2 tareas (Saudi PPO y RPPO) OK; las 48 tareas lanzadas en el cluster el 25-oct en la noche. Salida:
`trained_models/retrained_v9r2_startmix_20261025/`. Estado y pasos en la §7 de la propuesta. El mismo script de esta
carpeta es el control E1 sobre los agentes post-entrenados (`--agents trained_models/retrained_v9r2_startmix_20261025/seedXX`).

E1 sobre la prueba (`start_probe_v9r2sm_test_seed01_20261025.csv`, `.log`; Saudi, semilla 1, mismos 200 casos) [V]:

| Saudi | Modo | DSQ % original | DSQ % post-entrenado | Pos. media original | Pos. media post-entrenado |
|---|---|---|---|---|---|
| PPO | propia (C4) | 0,5 | 0 | 4,23 | 4,08 |
| PPO | C2 | 60 | 0 | 13,89 | 4,59 |
| PPO | C3 | 87 | 0 | 17,94 | 4,49 |
| RPPO | propia (C4) | 0 | 0 | 4,24 | 4,04 |
| RPPO | C2 | 0 | 0 | 4,74 | 4,62 |
| RPPO | C3 | 100 | 1 | 20,00 | 4,47 |

## 26-oct: E1 y E2 sobre las 48 réplicas (3 semillas x 8 GP x PPO/RPPO)

E1 [V]: `e1_20261026/` (`probe_seed<SS>_<GP>.{csv,log}`, todo junto en `e1_all_20261026.csv`); mismo script, 200 carreras
por modo, pareadas. Pasan el criterio (DSQ <= 1 % con todos los compuestos forzados) 19 de 48 réplicas; 11 de 16 agentes
tienen al menos una réplica que pasa. Antes del post-entrenamiento el máximo era 87-100 % en 14 de 16 agentes.
- Bahrain (PPO y RPPO, las 6 réplicas): 100 % DSQ con largada forzada C1, igual que el original; con C2 y C3, 0 %.
  Traza (seed01 PPO): tras largar en C1 la política para pidiendo C1 otra vez (acción 1), teniendo un C2 nuevo disponible.
  Las curvas de evaluación de Bahrain PPO quedan muy negativas durante todo el post-entrenamiento.
- Sin réplica que pase: Belgian RPPO (mín. 2 %), Dutch RPPO (1,5 %), US RPPO (3 %).
- Modo "propia": misma largada que el original en los 48 casos (100 % el mismo compuesto) y 0 % DSQ salvo Saudi (<= 1,5 %).

E2 [V]: `e2_selection_20261026/` (validación `val_*.csv` con `select_seeds_v5_20261010.py --calib v9r2 --seeds 1 2 3`,
F1_SELECT_TREE/OUT, reglas 5/7/8, `--max-dsq 0` como la versión del paper; resumen `seed_summary_r578_dsq0.csv`,
`select_pick.log`). Todas las semillas pasan la regla salvo Saudi PPO s3, Miami RPPO s3 y US PPO s2. La regla elige
copias en `trained_models/retrained_v9r2_startmix_20261025/selected/` (SIN filtrar por E1). Cruce con E1 y con los
agentes originales (misma validación): `e2_selection_20261026/e2_vs_e1_20261026.csv`. Posición media de validación,
promedio de los 16: original 6,601, elegidas por la regla 6,658 (PPO 6,627 -> 6,641; RPPO 6,574 -> 6,676); el original es
el mejor de 10 semillas y el post-entrenado el mejor de 3. 5 de las 16 elegidas por la regla pasan E1 (Saudi PPO y RPPO, Belgian PPO, Singapore PPO, US PPO).

**E1 en la largada real (decisión del autor, 26-oct)** [V]: como el análisis B solo pone al agente en la largada real
2024 del piloto, el criterio E1 se aplica solo a esa fila de la sonda (`e1_real_start_20261026.py` ->
`e1_real_start_20261026.csv` por réplica y `e2_real_start_20261026.csv` con la elección = regla del paper + E1 en la
largada real). Pasan 45 de 48 réplicas; las 3 que fallan son US RPPO (largada real C3, el agente larga C2): 32 %, 14,5 % y
3 % de DSQ. En los otros 15 agentes la réplica elegida por la regla ya pasa (0 % de DSQ, salvo Saudi RPPO con 1 %), así
que `selected/` queda tal cual. Largada real distinta de la propia del agente: Bahrain (C3 vs C2), Saudi (C3 vs C4) y
US RPPO (C3 vs C2). Bahrain con C1 (100 % DSQ) no aparece en el análisis B.

## 26-oct: E3, E4 y control de Saudi vuelta 7

**E4** [V]: `../decision_support_v9r2smrealhist_20261026/README.md`. Máscara de la acción recomendada 9,9 % -> 2,2 % en
promedio; Saudi 75 % -> <= 4,8 % (pasa); US RPPO 31 % -> 31 % (no pasa); rec = mejor 93,4 % -> 94,3 %.

**E3** [V]: Figs. 6–7 con los agentes post-entrenados y la QMIP (500 carreras), `e3_world2024_20261026/`
(`fig6_2024_realdnf/`, `fig7_2024/`, logs; comparación `e3_compare_20261026.csv`). QMIP idéntica a la original.
Promedio de 8 GP, posición media original -> post-entrenado:

| Escenario | PPO | RPPO | QMIP |
|---|---|---|---|
| (i) Fig. 6 | 7,800 -> 7,735 | 7,844 -> 7,860 | 8,136 |
| (ii) Fig. 7 | 8,616 -> 8,611 | 8,782 -> 8,593 | 9,100 |

- Criterio "no peor en más de 0,1": pasa (peor caso RPPO (i), +0,016). PPO y RPPO siguen por delante de la QMIP en ambos.
- Criterio "0 % DSQ": no se cumple estrictamente. Máximo 1,8 % (Saudi RPPO, (i)); Miami RPPO 1,2 % (i) y 1,0 % (ii);
  PPO <= 0,2 %. Los originales tenían hasta 0,6 % (Saudi PPO, (i)).
- Por GP, mayores empeoramientos: Singapore PPO (ii) +0,45, Saudi RPPO (i) +0,24, Belgian RPPO (i) +0,23; mayores
  mejoras: Dutch RPPO (ii) -0,59, Bahrain RPPO (ii) -0,36.

**Control de Saudi vuelta 7 (Figs. 8–10)** [V]: `saudi_lap7_20261026/` (`ds_rollouts_belief_20261012.py`, trayectoria
propia como el barrido v9r2, M = 500, vuelta 7). Mismo estado que el paper (P5 (i) / P6 (ii), C4, 9 vueltas), misma
recomendación (medios) y mismo orden de acciones (medios > duros > no parar > blandos en (ii)); posiciones medias a <= 0,2
de las del paper. Las Figs. 8–10 no cambian de conclusión con los agentes post-entrenados.
