# Saudi PPO, parada en la vuelta 49 (Fig. 7 v7) — diagnóstico 2026-10-17

Agente: PPO Saudi_Arabian_NOR, semilla 5 (elegida con `--ok-neutral`). En la Fig. 7 v7 para en la v7 (bajo el SC real
de 2024) y, en 154 de 500 carreras (30.8 %), otra vez en la v49 de 50.

Archivos: `diag_20261017.py` (envoltorio de `scripts/evaluate_world2024_20261010.py`; `--mode trace` registra el estado
del agente desde la v40, `--mode nolate` prohíbe parar desde la v45), `trace/`, `nolate/` (Fig. 7 Saudi, solo PPO +
QMIP, 500 carreras, semilla 0), `seeds/seedXX/` (Fig. 7 Saudi con cada una de las 10 semillas de PPO).

Resultados [V]:
- `trace` reproduce exactamente la Fig. 7 (posiciones y vueltas de parada iguales en las 500 carreras).
- La parada de la v49 es a un neumático nuevo del **mismo compuesto** (acción 2, compuesto 3, TyreLife 39), sin SC/VSC,
  con 2 rivales retirados (los reales de 2024, como en todas las carreras). Las carreras con parada tienen al agente algo
  más adelante (posición media 4.94 vs 5.99) y con más margen atrás (9.7 vs 7.7 s), pero los rangos se solapan:
  no hay una condición que la justifique (la pérdida de la parada es mucho mayor que ese margen).
- Contrafactual `nolate` (mismas carreras, sin parar desde la v45): en esas 154 carreras la posición media pasa de 6.64
  a 4.95 (mejora en 142, igual en 12, nunca empeora); en las otras 346 nada cambia. Media de PPO en Saudi: 6.20 → 5.68.
- Las otras 9 semillas de PPO en la misma Fig. 7: 0 % de paradas tardías (una semilla 0.4 %), posición media 5.36–5.72
  (la semilla 8, que elegía el filtro original, 5.68). Es un rasgo de la semilla 5.
- En la validación de la selección la semilla 5 no se distingue: 8 % de carreras con parada en las últimas 6 vueltas,
  igual que la 7 y la 8 (la 9 tiene 17.5 %); correlación entre ese % de validación y el de la Fig. 7: 0.06.

Interpretación (hipótesis): la Fig. 7 de Saudi es muy homogénea (todos los rivales paran en la v7 bajo SC, así que en la
v49 todas las carreras están en un estado parecido: duros con 39 vueltas, sin SC), y la política de la semilla 5 tiene
una región pequeña en que elige parar; la Fig. 7 cae en ella una y otra vez. No lo causa el filtro nuevo en sí, pero
coincidió que este cambio de filtro eligió la semilla 5.
