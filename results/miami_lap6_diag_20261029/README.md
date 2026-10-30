# Diagnóstico C: PPO/RPPO paran en la vuelta 6 en Miami (v10eGQf, 29-oct-2026)

Contexto: `docs/PROPUESTA_FIGS67_V10F_20261029.md` §1 (raíz del proyecto). Todo lo de abajo está **verificado** (leído en
datos o corrido), salvo lo marcado como hipótesis. Mundo = el de la selección de semillas (= entrenamiento: rivales
estocásticos, abandonos v6, SC/VSC sorteados con la semilla 2026, `env_seleccion_20261029.sh` + envoltorios v10f).

## 1. Solo pasa en Miami, y solo en PPO/RPPO (`primera_parada_20261029.py`, CSV de validación de la selección)

- Primera parada en la vuelta 6: PPO 80 % y RPPO 72 % de las 2000 carreras de validación, en Miami. TRPO tiene 17 %.
  En los otros 7 GP, ningún método pasa del 2 %.
- Semillas que paran en la 6 en más de la mitad de las carreras: PPO 9/10, RPPO 8/10, TRPO 2/10. Ninguna en otro GP.

## 2. Curvas de entrenamiento (`curvas_eval_20261029.py`, `evaluations.npz`)

- Miami es el único GP donde la mejor recompensa de evaluación de PPO/RPPO (−4145 / −4128, media de 10 semillas) queda
  **peor** que la de DQN/A2C (−3860 / −3909). En los otros 7 GP, PPO/RPPO son los mejores.
- Al millón de pasos, PPO/RPPO en Miami ya están en ≈ −4300 y casi no mejoran después. DQN/A2C siguen en ≈ −6450
  (nivel de descalificación) y suben más tarde.

## 3. Entropía (`entropia_20261029.py`, logs)

- PPO/RPPO colapsan la entropía temprano en **todos** los GP: ≈ 0,1 a los 250 mil pasos (máximo ln 4 = 1,39). Esto pasa
  pese a `ent_coef` 0,02 en PPO.
- En Miami el colapso es el más temprano y el más bajo: entropía < 0,01 a los 0,73 M (PPO) y 0,42 M (RPPO) pasos.
  Saudi y Singapore se parecen (0,6–0,7 M).

## 4. Estrategias fijas de una parada (`fixed_sweep_20261029.py`, 200 carreras, mismos escenarios en todas)

90 estrategias: arranque s, parada en la vuelta L, compuesto c (C2/C3/C4). Si la parada cae en la 2.ª vuelta o una
posterior de un SC, no tiene efecto (regla `sc_bunching`), así que se hace en la siguiente vuelta válida.

| | Recompensa media | Posición media |
|---|---|---|
| Parada en la vuelta 6 (mejor de las 6 combinaciones) | −4225 | 13,2 |
| Mejor fija por recompensa (C3, vuelta 33, C4) | −4041 | 12,6 |
| Mejor fija por posición (C3, vuelta 21, C4) | −4100 | 12,5 |
| PPO s1 (elegida) | −4264 | 13,4 |
| RPPO s1 (elegida) | −4275 | 13,4 |
| A2C s8 (elegida) | −3949 | 12,2 |
| DQN s8 (elegida) | **−3630** | **11,3** |

- **El paisaje de estrategias fijas es plano.** Entre las vueltas 7 y 36, la mejor combinación por vuelta varía solo
  entre −4041 y −4162 de recompensa y entre 12,5 y 12,9 de posición. La vuelta 6 es la peor de ese tramo, pero por poco:
  la mejor fija gana +193 ± 94 de recompensa y −0,59 ± 0,36 posiciones (pareado, IC 95 %).
- PPO/RPPO reproducen la parada fija en la vuelta 6 (C3 → C2), con el mismo resultado.
- **DQN gana por reaccionar a SC/VSC** (`paradas_sc_20261029.py`). El 55 % de sus paradas cae en una vuelta neutralizada
  o junto a ella, contra 13 % de PPO/RPPO y 19 % de A2C. En el 70 % de las carreras hay SC/VSC después de la vuelta 6;
  ahí DQN queda en 10,9, contra 12,8 de la mejor fija y 13,5 de PPO/RPPO. Sin SC/VSC, DQN queda en 12,2, igual que las
  fijas.

## 5. Lectura

- **Verificado:** no es un error del entorno. La vuelta 6 no está premiada: es una estrategia fija algo peor que
  las demás. Lo que vale en Miami es una política que **espera y para bajo SC/VSC**, y DQN la aprende.
- **Hipótesis:** PPO/RPPO colapsan su política antes de descubrir esa política, porque al principio la señal dominante es
  evitar la descalificación (−5000). "Parar en cuanto no hay penalización" la evita siempre, y entre estrategias fijas
  casi no hay gradiente que la saque de ahí. En los otros GP el paisaje es menos plano o el SC pesa menos, y el mismo
  colapso cae en una buena estrategia.
- **Hipótesis no verificada:** por qué se volvió dominante en v10f, si en v9r2 PPO paraba en la 6 en 76 de 2000
  carreras. Candidatos: el adelantamiento v10f (más adelantamientos, menos diferencia entre estrategias fijas) o la edad
  efectiva v10eGQ.

## 6. Opciones para B (requieren acuerdo; entrenamiento)

1. **Reentrenar PPO/RPPO en los 8 GP con más exploración**, por ejemplo `ent_coef` 0,05 en PPO. RPPO usa su propio control de entropía (`scripts/train_rl_agents.py` l. 376–430), que habría que subir de forma análoga.
   Es el criterio uniforme. Costo: 160 tareas de ~1,5 h en el cluster, y repetir la selección y las Figs. 6–7.
2. **Solo Miami con más exploración** (20 tareas). Es más barato, pero es una elección hecha a mano para un GP. Habría
   que justificarla en el texto con el diagnóstico de arriba.
3. **Prueba previa barata:** 2 semillas de PPO Miami con `ent_coef` 0,05, a 1,5 M pasos en el Mac (~30 min en
   paralelo), para ver si escapan de la vuelta 6 antes de gastar en el cluster.

Reproducir: `bash results/miami_lap6_diag_20261029/lanzar_20261029.sh` desde la raíz del repo (~12 min con 7 procesos)
y luego los scripts de análisis. Datos en `data/`.

## 7. Prueba B3: PPO Miami con más entropía, 1,5 M pasos en el Mac (29-oct, noche)

Lanzador `b3_lanzar_20261029.sh` (mismo comando que el cluster, salvo `--ent-coef`, `--timesteps 1500000` y
`--out-root`). Agentes en `trained_models/probe_miami_entropy_20261029/`. Evaluación en `b3_evaluar_20261029.sh`, con las
mismas 200 carreras de la §4; resultados en `b3/`.

| `ent_coef`, semilla | Mejor eval. en el entrenamiento | Recompensa (200 carreras) | Posición | Primera parada en la v. 6 | Patrón más común |
|---|---|---|---|---|---|
| 0,05, s1 | −5144 | −5148 | 17,5 | 84 % | C2, v6→C3, v8/9→C3 |
| 0,05, s2 | −5173 | −5161 | 17,5 | 72 % | C2, v6→C2, v8→C2 |
| 0,10, s1 | −4539 (a los 0,15 M) | −4496 | 14,3 | 51 % | C3, v6→C4 |
| 0,10, s2 | −5111 | −5117 | 17,4 | 50 % | C3, v4→C2, v7→C2 |
| original 0,02, s1 (5 M) | −4173 | −4264 | 13,4 | 89 % | C3, v6→C2 |

- **Verificado:** con más entropía, la entropía sí se mantiene alta (0,65–0,73 al 0,9 M, contra 0,015 de la original).
  Pero la parada en la vuelta 6 **sigue** siendo la primera en el 50–84 % de las carreras, y además aparece una
  segunda parada temprana (vueltas 7–10). Al 1,5 M los agentes son peores que la original. Más entropía, por sí sola,
  no saca a PPO de la parada temprana en este horizonte. No se probó a 5 M.
- **Hipótesis (no verificada):** el atractor viene de la estructura de acciones. 3 de las 4 acciones son "parar", así
  que una política casi uniforme para en casi todas las vueltas. Aprender a no parar durante 20–30 vueltas seguidas,
  esperando un SC, exige muchas decisiones consecutivas iguales. Para un método on-policy, la frontera natural es la
  primera vuelta sin penalización. DQN parte de otro lado: con ε-greedy en torno a un Q aprendido fuera de la política.

## 8. Opción 1: las mismas 4 pruebas a 5 M pasos (29-oct, noche)

`b3_lanzar_5M_20261029.sh` y `b3_evaluar_5M_20261029.sh`. Agentes en `trained_models/probe_miami_entropy5M_20261029/`,
resultados en `b3_5M/`. Mismas 200 carreras que en la §4.

| `ent_coef`, semilla | Mejor eval. (paso) | Recompensa | Posición | DSQ | Primera parada en la v. 6 | Entropía final |
|---|---|---|---|---|---|---|
| **0,05, s1** | **−3692 (4,95 M)** | **−3720** | **11,46** | 2 % | 3 % | 0,017 |
| 0,05, s2 | −4656 (3,9 M) | −4604 | 14,49 | 15 % | 66 % | 0,65 |
| 0,10, s1 | −4539 (0,15 M) | −4496 | 14,32 | 6 % | 51 % | 0,75 |
| 0,10, s2 | −4502 (2,55 M) | −4449 | 14,35 | 2 % | 100 % | 0,77 |
| original 0,02 (10 semillas) | −4145 (media) | | 13,4 (s1) | | 9/10 semillas | ≈ 0,001 |
| DQN s8 (elegida) | | −3630 | 11,26 | | 0 % | |

- **Verificado:** con `ent_coef` 0,05, la semilla 1 sale de la parada temprana hacia los 2,5 M pasos. Su evaluación pasa
  de ≈ −6400 a −3968 y sigue subiendo hasta −3692 a los 4,95 M. Llega al nivel de DQN (11,46 contra 11,26) y aprende lo
  mismo que DQN: el 55 % de sus paradas cae junto a un SC/VSC, igual que DQN. En las carreras con SC/VSC después de la
  vuelta 6 queda en 11,27 (DQN 10,87).
- Con 0,10 la entropía no baja (0,75–0,77 al final) y la política determinista no se ordena. Con 0,05, una de dos
  semillas escapa. Con la original 0,02, ninguna de 10.
- **Lectura:** el problema de Miami es de exploración. Un `ent_coef` algo mayor lo corrige en parte: con 10 semillas y la
  selección habitual, es probable que haya semillas buenas. Esto es una inferencia a partir de 2 semillas.

## 9. Reentrenamiento con `ent_coef` 0,05 en el cluster (29-oct, OK del autor: "B1 para PPO" + prueba análoga en RPPO)

- Tareas: `reproduce/cluster/make_tasks_v10eGQf_ent05_20261029.py`, derivadas de `tasks_v10eGQf_20261029.tsv`. Usan el
  mismo comando; solo cambia la entropía y el árbol de salida.
  - **PPO**: `tasks_v10eGQf_ppoent05_20261029.tsv`, 8 GP × 10 semillas, `--ent-coef 0.05`, salida en
    `trained_models/retrained_v10eGQf_ppoent05_20261029/`.
  - **RPPO**: `tasks_v10eGQf_rppoent05_20261029.tsv`, solo Miami, 10 semillas, `--hyperparams rppoent05_20261029`
    (`hyperparams/config_rppoent05_20261029.py` = v1 con `ent_coef` 0,05), salida en
    `trained_models/retrained_v10eGQf_rppoent05_20261029/`.
    Si funciona en Miami, se extiende a los 8 GP.
- Prueba pequeña:
  - trabajos 18083095 (PPO, tarea 22 = Miami s1) y 18083096 (RPPO, tarea 1 = Miami s1);
  - los logs muestran `[v10f]`, `ent_coef=0.05` efectivo y ~900 fps;
  - MaxRSS a los 10 min: 0,44 GB (PPO) y 0,62 GB (RPPO).
- Arreglo completo: 18083158 (PPO, tareas 1-21 y 23-80, `%20`) y 18083159 (RPPO, 2-10), con 830M.
- Después:
  - bajar con rsync, sin `--delete`;
  - evaluar Miami como en `b3_evaluar_5M_20261029.sh` (primera parada en la vuelta 6, posición, mismas 200 carreras);
  - rehacer la selección y las Figs. 6–7 con PPO (y RPPO) nuevos.

## 10. Resultado RPPO Miami con `ent_coef` 0,05 (cluster 18083159 + 18083096, 30-oct madrugada)

- Las 10 tareas se completaron en 1,7–1,9 h cada una, con MaxRSS de 0,59–0,74 GB.
- Los agentes se bajaron con rsync, sin `--delete`, a `trained_models/retrained_v10eGQf_rppoent05_20261029/`.
- Evaluación: `ent05_evaluar_20261029.sh` con las mismas 200 carreras de la §4. Resumen con
  `ent05_resumen_20261029.py` (correr tras `source env_seleccion_20261029.sh`). La línea base son los agentes originales
  con `ent_coef` 0,02, pasados por el mismo evaluador (`ent05/base_*`).

| | Posición media (10 semillas) | Mejor semilla | Semillas con 1.ª parada en la v. 6 < 20 % | Semillas con pos. < 12 |
|---|---|---|---|---|
| RPPO original (0,02) | 13,46 | 13,19 | 2/10 | 0/10 |
| **RPPO 0,05** | **12,14** | **11,41** (s9, DSQ 0) | **7/10** | **7/10** |
| PPO original (0,02) | 13,44 | 13,37 | 0/10 | 0/10 |
| DQN s8 (elegida, §4) | 11,26 | | | |

- **Verificado:**
  - Las 7 semillas que escapan quedan en 11,41–11,74. Entre el 38 % y el 56 % de sus paradas cae junto a un SC/VSC
    (las originales, 6–28 %). Es la misma política reactiva de DQN.
  - Las otras 3 semillas (1, 3, 7) siguen parando en la vuelta 6, en 13,3–13,5.
  - La mejor evaluación del entrenamiento sube de −3991 a −4246 (originales) a −3537 a −3665 en las 7 que escapan.
  - La entropía se mantiene en ≈ 0,4–0,7 durante el entrenamiento (s9).
- **Lectura:**
  - Con `ent_coef` 0,05, RPPO corrige Miami en la mayoría de las semillas, y la selección habitual elegiría una que
    escapa.
  - Falta ver si el cambio empeora a RPPO en los otros 7 GP. Para eso habría que reentrenarlo en los 8 GP (70 tareas
    más), como PPO. Requiere OK del autor.
- **Selección de semillas** (`ent05_seleccion_20261030.sh`, mismas reglas que `pipeline_v10`, salida en
  `results/seed_selection_v5_20261010/v10f_rppoent05/`):
  - elige la **semilla 9, con 11,41** (DSQ 0); la RPPO elegida hoy en Miami queda en 13,4;
  - sus cifras por semilla coinciden exactamente con las del evaluador de esta sección.
