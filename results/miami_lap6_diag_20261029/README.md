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
