# Validación predictiva con la estrategia real 2024 (27-oct-2026)

El piloto del agente ejecuta su largada y sus paradas reales de 2024 durante toda la carrera. Se usa el mismo entorno,
mundo 2024 (v7) y semillas que las Figs. 6–7 del paper (`dsq0`). En el escenario `fig7`, los rivales siguen sus
estrategias reales y las banderas y abandonos son los reales; en `fig6`, los rivales van por logit, las banderas se
sortean y los abandonos son los reales. Se compara con la clasificación real en pista: última vuelta de
`data/session_2024_V2.csv`, con los reemplazos BEA→SAI, COL→SAR y LAW→RIC. Son 500 carreras por GP y escenario; corre en
unos 3 min en el Mac con 8 procesos.

- Corridas: `run_one_20261027.sh <fig6|fig7> <GP_DRV>` (entorno en `env_20261027.sh`; script
  `scripts/validate_real_strategy_20261027.py`). Salida: `fig6/`, `fig7/` (posición de todos los pilotos por carrera).
- Comparación: `compare_20261027.py`, que escribe `compare_{agent,all,summary}_20261027.csv`.

Resultados [V]:
- Piloto del agente: la posición real cae dentro del intervalo simulado del 90 % en 16 de 16 casos (8 GP × 2 escenarios).
  En `fig7`, la mediana simulada es mejor que la real en 6 de 8 GP. La media simulada es 9,40 y la real 10,25, lo que
  indica un leve optimismo para estos pilotos.
- Todos los pilotos que terminaron (152 por escenario): la cobertura del intervalo del 90 % es 88,8 % (`fig7`) y 94,1 %
  (`fig6`); el Spearman medio entre la media simulada y la posición real es 0,91 y 0,88.
- Con las mismas semillas, la estrategia real simulada da una posición media de 9,40 en `fig7` y 9,51 en `fig6`. PPO da
  8,62 y 7,80; la QMIP, 9,10 y 8,14. En `fig7`, la estrategia real es mejor que todos los métodos en Dutch y Singapore.

En el paper (27-oct, ítems 1–3 aprobados): la Fig. 7 incluye la caja gris "Actual" (`results/world2024_v9r2_20261023/fig7_actual_20261027.py`).
El párrafo de la Fig. 7 compara los métodos con la estrategia real simulada: A2C −0,83, PPO −0,79, RPPO −0,62, QMIP −0,30
y TRPO −0,17; DQN +0,06, no significativo; en Dutch y Singapore la estrategia real es mejor que todos los métodos. También
agrega la validación (8/8 dentro del intervalo del 90 %, cobertura de 89 % y Spearman de 0,91 para todos los pilotos, y
0,85 posiciones de optimismo para el piloto del agente). La Fig. 6 y la Tabla O.1 no cambian.
