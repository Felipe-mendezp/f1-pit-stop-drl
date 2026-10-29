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
