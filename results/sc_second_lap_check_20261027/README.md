# Chequeo (b): ¿qué pasa si el agente puede parar también en la 2.ª vuelta del SC? (27-oct-2026)

Con `sc_bunching` (activo en el paper), el agente y los rivales solo pueden parar en la primera vuelta de un período de SC.
Este chequeo levanta esa restricción **solo para el agente** y **solo en los rollouts del look-ahead**. Para eso usa el
atributo opcional `sc_pit_max_idx` de `src/environment/f1_env_all_drivers.py`, que por defecto vale 1 y deja el entorno
idéntico. A las 4 acciones de siempre se agregan tres brazos: "no parar en la vuelta n y parar en n+1 al compuesto x";
después decide la política. Los rivales logit no cambian.

Script: `scripts/ds_sc_second_lap_check_20261027.py`. Lanzador: `run_20261027.sh <fig8|fig11> <GP> <Driver> <ModelDir>
<vuelta> <k> <M> <tag>`. Resumen: `summary_20261027.py` → `summary_20261027.csv`. Logs en `logs/`. Cada caso con
M = 500 tardó ≈ 2–3 min en un núcleo.

- `fig8/`: misma configuración que las Figs. 8–10 (agentes `rl_agents_v9r2_20261022/selected`, `ds_rollouts_belief`).
- `fig11/`: misma configuración que la Fig. 11 (agentes post-entrenados, `F1_DS_FORCE=hist`).

## Controles [V]

- Con k = 1 (la regla del paper), los rollouts coinciden bit a bit con los CSV publicados (M = 30): Fig. 8 Saudi v. 7 y
  Fig. 11 Saudi v. 7.
- Con k = 2, los 4 brazos de siempre también coinciden bit a bit con los publicados (M = 500). La política nunca elige
  parar en la vuelta n+1 por sí sola. Solo cambian los brazos nuevos.

## Resultados [V]

Puntos medios. "Δ" es la diferencia pareada frente a la mejor acción del paper (±2 SE).

| Caso | Mejor acción del paper | No parar en n (Δ) | Mejor brazo "parar en n+1" (Δ) |
|---|---|---|---|
| Fig. 8, Saudi v. 7 | Medio, 12,95 | −1,46 ± 0,70 | Medio en v. 8: **+0,75 ± 0,67** |
| Fig. 11, Saudi v. 7 | Blando, 13,79 | −2,70 ± 0,69 | Blando en v. 8: **+0,12 ± 0,70** |
| Fig. 11, Miami v. 28 | Blando, 1,72 | −0,43 ± 0,20 | Blando en v. 29: −0,04 ± 0,21 |
| Fig. 11, US v. 3 | No parar, 5,14 | 0 | Medio en v. 4: +0,19 ± 0,32 |

## Lectura

1. **Si se permite parar en la 2.ª vuelta del SC, parar ahí rinde igual que parar en la 1.ª.** La diferencia no es
   significativa en ningún caso. La conclusión "conviene parar bajo este SC" se mantiene. En cambio, la vuelta exacta
   (7 u 8 en Saudi) no se puede separar.
2. **La cifra "quedarse afuera pierde 1,46 puntos" (Fig. 8) depende de la regla.** Mide quedarse afuera **sin** la opción
   de parar en la vuelta 8. Si existiera esa opción, quedarse afuera en la vuelta 7 y parar en la 8 no sería peor.
3. **Fig. 11 Saudi (+2,7 puntos) no cambia en lo esencial.** Compara parar en la vuelta 7 con lo que hizo Norris, que se
   quedó afuera durante todo el SC; en 2024 nadie paró en las vueltas 8 ni 9. Lo que el chequeo matiza es que también
   habría bastado parar en la vuelta 8.
4. **Miami y US:** sin cambios en las conclusiones.

[H] La parada en la 2.ª vuelta puede salir algo favorecida por la mecánica del modelo. En esa vuelta los gaps se reducen a
la mitad después de aplicar el PitIn, y en la 3.ª la fila borra la vuelta de salida. Por eso la ventaja de +0,75 en la
Fig. 8 no debe leerse como "mejor parar en la 8".

## ¿En qué vuelta del SC paran los equipos? [V]

`sc_stop_lap_share_20261027.py`, 2021–2024, carreras secas y sin bandera roja: 258 paradas bajo SC. De ellas, 173 (67 %)
fueron en la 1.ª vuelta del período, 60 (23 %) en la 2.ª y 25 (10 %) en la 3.ª o después. La vuelta de SC se define por
el TrackStatus mayoritario, así que algunas paradas "en la 2.ª" pueden ser la primera oportunidad de quien ya había pasado
la entrada a boxes [H]. La bandera `PitIn` no sirve para esto, porque bajo SC cuenta de más.
