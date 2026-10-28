# Cuándo la recomendación es casi indiferente (27-oct-2026, análisis 3)

Se calcula con `ambiguity_20261027.py`, que escribe `ambiguity_laps_20261027.csv` y `ambiguity_summary_20261027.csv`. Usa
los rollouts del barrido E4 (PPO post-entrenado, escenario `real` = Fig. 11, 460 vueltas de 8 GP, M = 500, pareados por
rollout).

Métricas por vuelta, en puntos:
- **Decisión de parar:** no parar frente a la mejor parada, ±2 SE.
- **Compuesto:** la mejor parada frente a la segunda.
- **Mejor frente a segunda mejor acción:** también se calcula, pero es poco informativa, porque "no parar" gana con claridad
  casi siempre.

Resultados [V]:
- **Quedarse afuera es significativamente mejor** que cualquier parada en 388 de 460 vueltas.
- **Parar es significativamente mejor** en 17 vueltas: Bahrain 13–15, 34–37 y 40–41; Saudi 7; Miami 28; US 32, 34–38.
- **La diferencia queda dentro de ±2 SE** en 55 vueltas.
  - En 26 de ellas no hay puntos en juego: todos los rollouts dan la misma diferencia, 0. Son Miami (12, de ellas 29–32
    bajo SC), Belgian (10), Saudi 8–9 y US 4–5 (vueltas 2+ de SC, donde no se puede parar; ver abajo).
  - En las otras 29, la diferencia en posición tampoco es significativa en 19.
- **Compuesto:** dentro de las 72 vueltas de "ventana" (parar mejor o indistinguible), el mejor compuesto no se separa del
  segundo en 53.

**Resuelto (27-oct): no es un bug.** En Saudi 8–9, US 4–5 y Miami 29–32 [V] las cuatro acciones dan resultados idénticos
porque son vueltas 2+ de un período de SC, y con `F1_SC_BUNCHING=1` (`reproduce/env_paper.sh`) el entorno convierte la
parada del agente en no-op sin castigo (`f1_env_all_drivers.py`, paso 1 de `step`). Es la regla del modelo, descrita en el
paper (App. de evolución de la carrera: "pit stops are only possible on the first lap of the period"; §7: "On laps 8 and 9
the four actions coincide"). Bajo VSC el agente sí puede parar en cualquier vuelta; los rivales, solo en la primera.

**Cifras sin esas 8 vueltas, donde no hay decisión [V]:** 452 vueltas; quedarse afuera significativamente mejor en 388,
parar en 17, indistinguibles 47 (18 sin puntos en juego: Belgian 10, Miami 8; de las 29 restantes, posición también
indistinguible en 19); ventana = 64 vueltas, compuesto indiferente en 45.
