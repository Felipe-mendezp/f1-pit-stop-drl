# Evaluación de modelos de rivales (2026-10-14)

Pregunta: ¿el modelo de paradas de los rivales del simulador (S0 + reglas + K0) es razonable, o conviene un v7 con el
modelo del paper o una mezcla? Ajuste con 2021–23 y prueba con 2024, sin entrenar agentes.

- `reproduce_check_20261014.py`: reajustar S0 y K0 con `src/models/rival_models.py` reproduce los pickles del simulador
  (diferencia máxima 3e-15; iguales en `simulation/` y `simulation_v6_20261014/`).
- `rivals_common_20261014.py`: datos, modelos S0–S3 y K0–K1, reglas R (simulador), R2 (mínima) y R3 (R en verde,
  libre bajo SC/VSC).
- `eval_lap_20261014.py` (primera versión) y `eval_lap_v2_20261014.py`: evaluación vuelta a vuelta. En v2, cada piloto ×
  periodo de SC/VSC es una unidad (criterio del autor: parar en cualquier vuelta del periodo cuenta como acierto si el
  modelo para en la primera; vale para SC y VSC).
- `sim_strategies_20261014.py`: estrategias completas simuladas (200 por piloto, 8 GP del paper) frente a las reales
  de 2024, con SC/VSC y posiciones reales como exógenos.

## Resultados

- Etapa 1 (vuelta a vuelta): S1 (paper, LapsLeft lineal) es el peor. S0, S2 y S3 casi iguales. Las reglas del simulador
  prohíben el 9.3–9.9 % de las paradas reales de 2024 (sobre todo al final de la carrera en verde), y empeoran el
  log-loss.
- Etapa 2: el logit condicional (K1) predice mejor que el binario normalizado (K0): log-loss 0.710 vs 0.731 (22 GP),
  0.758 vs 0.794 (8 GP).
- Estrategias completas: las reglas de fin de carrera son lo que más importa. Sin ellas (B, C, D) salen ~2.0 paradas
  por piloto (real 1.50); con ellas (A, E, G) ~1.67–1.68. El simulador actual (A) está entre los mejores; cambiar K0
  por K1 (G) o S0 por S3 (E) casi no cambia las estrategias. Ningún modelo reproduce bien la vuelta de la primera
  parada (KS ≈ 0.36–0.40) ni la proporción de una parada (real 55 %, modelos ≤ 42 %), en parte por el cambio de 2024
  (1.52 paradas por piloto frente a 1.65 en 2021–23).
- Entre 8 y 11 % de las estrategias simuladas terminan sin dos compuestos: esta simulación no aplica la regla del
  simulador que fuerza otro compuesto (`_ensure_compound_change`).

Conclusión: ningún candidato mejora de forma clara al modelo actual a nivel de estrategias; el del paper es el peor.

## 2026-10-15: reescalamiento de TyreLife 2021
`rivals_common_20261014.py` acepta `TLV` (por defecto `vigente`, que reproduce lo anterior), y `sim_strategies_20261014.py`
acepta `COMBOS` (subconjunto por letra). Las salidas de otras variantes llevan el sufijo `_tl<variante>`. Resultados y
conclusión en `../tyrelife_2021_20261015/README.md`.
