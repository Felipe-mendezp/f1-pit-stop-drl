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
  - En 26 de ellas no hay puntos en juego: todos los rollouts dan la misma diferencia, 0. Son Miami (12), Belgian (10),
    Saudi 8–9 y US 4–5.
  - En las otras 29, la diferencia en posición tampoco es significativa en 19.
- **Compuesto:** dentro de las 72 vueltas de "ventana" (parar mejor o indistinguible), el mejor compuesto no se separa del
  segundo en 53.

**Pendiente, hallazgo a investigar.** En Saudi 8 y US 4 [V] las cuatro acciones dan resultados idénticos en los 500
rollouts (la misma media de puntos y de posición, y la parada siguiente en la misma vuelta: 42 y 22). La única diferencia
es que el script cuenta una parada más. Es decir, la parada forzada en esa vuelta no tiene efecto en el entorno. Son vueltas
de SC (la segunda vuelta de SC en Saudi y la segunda en US).

[H] Puede ser que el entorno ignore una parada en ciertas vueltas de SC, por ejemplo cuando el SC se "revela" con una
vuelta de anticipación, o que haya una regla de pit lane cerrado. Hay que revisarlo en `src/environment/f1_env_all_drivers.py`
antes de usar estas vueltas. No afecta a la Fig. 11 (en esas vueltas la recomendación coincide con lo que hizo el piloto) ni
a las Figs. 8–10 (vuelta 7). Sí podría afectar a los rollouts de otras vueltas bajo SC.
