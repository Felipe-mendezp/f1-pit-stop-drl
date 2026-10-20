# Reescalamiento de TyreLife 2021 (2026-10-15)

Pregunta: ¿están bien los factores con que se reescala la vida del neumático de 2021 para dejarla equivalente a 2022-23?
Código vigente: `TYRELIFE_ADJUST = {C0 .65, C1 1.0, C2 .90, C3 .98, C4 .92, C5 .92}` en `src/models/rival_models.py`
(rivales). Los mismos valores están copiados en `overtake_models.py`, `logit_overtakes_pit.py`, `regressions.py` y
`scripts/train_yf_models.py`. Origen: `Codigos/respaldo/30-09-2026 MAC Charles/tires_2021.py`.

## Revisión de tires_2021.py
- Incluye el último stint de cada piloto, que termina con la bandera a cuadros y no por desgaste (censurado).
- Compara medias simples por compuesto. Cada compuesto se usa en GP distintos según el año, así que la mezcla de
  circuitos cambia entre años.
- El largo se mide en vueltas contadas del stint, no en `TyreLife` (que es lo que se reescala). Además, `PitIn.cumsum()`
  asigna la vuelta de entrada al stint siguiente. Compara con 2022-24, no con 2022-23 (los años de ajuste).
- Los valores del código no coinciden exactamente con los que da el script (p. ej. C3: 0.96 con el script y 0.98 en el
  código; C5: 0.79 y 0.92).

## Métodos (`stint_factors_20261015.py`)
Stints terminados en parada en verde (vuelta de entrada y la anterior sin SC/VSC/roja), de 8 vueltas o más, sin
neumáticos de lluvia. Largo = `TyreLife` en la vuelta de entrada.
- M2: log(TyreLife al parar) ~ celda GP×compuesto + 2021×compuesto, solo celdas con datos en 2021 y en 2022-23, error
  estándar por GP.
- Dentro del logit de paradas (variante `inter`): sin reescalar, con pendiente de `TyreLife` propia de 2021 por
  compuesto. El cociente de pendientes 2021 / 2022-23 es el factor implícito.

| Compuesto | Vigente | M2 (stints) | Logit (inter) |
|---|---|---|---|
| C0 | 0.65 | 0.58 (1 celda, 3 stints) | 0.56 |
| C2 | 0.90 | 0.86 (0.76–0.98) | 0.88 |
| C3 | 0.98 | 0.86 (0.76–0.97) | 0.83 |
| C4 | 0.92 | 0.70 (0.57–0.86) | 0.75 |
| C5 | 0.92 | 0.785 (2 celdas) | 0.72 |
| Común | | 0.84 (0.76–0.93) | |

En el mismo circuito y compuesto, en 2021 se paró con un neumático 19 % más viejo que en 2022-23. En 2024 fue un 5 %
más viejo, sin diferencia significativa (IC 0.97–1.13).

## Efecto en el modelo de rivales (variantes `TLV` en `../rival_model_eval_20261014/rivals_common_20261014.py`)
`vigente`, `ninguno`, `comun` (0.84), `M2`, `inter` y `sin2021` (ajuste solo con 2022-23). Logs: `lap_*.log`, `sim_*.log`.
CSV: `../rival_model_eval_20261014/*_tl<variante>.csv`. La variante `vigente` reproduce exactamente la evaluación del 14-oct.

| Variante | AIC S0 2021-23 | log-loss S0+R 2024 (22 GP / 8 GP) | Paradas por piloto A (real 1.50) | Error en paradas A |
|---|---|---|---|---|
| vigente | 14349 | 118.5 / 106.3 | 1.675 | 0.272 |
| ninguno | 14396 | 117.8 / 106.0 | 1.663 | 0.265 |
| comun 0.84 | 14314 | 119.8 / 106.6 | 1.702 | 0.289 |
| M2 | 14310 | 119.8 / 106.5 | 1.699 | 0.288 |
| inter | 14308 | 120.0 / 106.3 | 1.697 | 0.282 |
| sin2021 | (otra muestra) | 121.3 / 105.3 | 1.718 | 0.289 |

- Dentro de 2021-23, corregir más (0.84) ajusta mejor que el vigente (AIC −35 a −41).
- En 2024 el efecto es el contrario y pequeño: los stints de 2024 son algo más largos que los de 2022-23, y los factores
  más fuertes suben las paradas previstas.
- En las estrategias simuladas, todas las variantes quedan entre 1.66 y 1.72 paradas por piloto. La diferencia es mucho
  menor que la distancia a la real (1.50). El reescalamiento no explica el exceso de paradas de los rivales.

## Fase 0 (15-oct): una sola fuente del factor
`src/tyrelife_2021.py` (`factors()`, `adjust_2021()`, variable `F1_TYRELIFE_2021`: `legacy` por defecto, o
`data/tyrelife_2021_factors_<tag>.json`). Reemplaza las 5 copias (`regressions.py`, `rival_models.py`,
`overtake_models.py`, `logit_overtakes_pit.py`, `scripts/train_yf_models.py`). Verificado con `legacy`: TyreLife de 2021
idéntico en los tres cargadores y pickles S0/K0 del simulador reproducidos (diferencia máx. 3e-15). El entorno y el
entrenamiento no importan estos módulos (nada que sincronizar al cluster). `data/tyrelife_2021_factors_none.json` = sin
reescalar.

## Fase 1 (15-oct): estimación con tiempos de vuelta (`laptime_factor_20261015.py`, log y CSV)
Ec. 1 por GP con la misma muestra del simulador (all_laps + filtro A + IQR), pendiente de 2021 = f × la de 2022-23, f común
a los 15 GP con vueltas de 2021 (15.537 vueltas), error estándar agrupado por carrera.

| Estimación | f | IC 95 % |
|---|---|---|
| Común (especificación del simulador) | 0.555 | 0.41–0.70 |
| Por compuesto: C2 / C3 / C4 / C5 | 0.55 / 0.54 / 0.55 / 0.60 | ancho en C5 |
| Por compuesto: C0 (1.240 vueltas) | 0.89 | 0.64–1.13 |
| Común, con LapNumber y LapNumber² por año (robustez) | 0.66 | 0.52–0.80 |
| 2023 relativo a 2022 (homogeneidad de la referencia) | 1.04 | 0.85–1.22 |

Validación 2024 (pendiente de TyreLife 2021-23 frente a la Ec. 1 de 2024 sola, 43 celdas GP × compuesto, s/vuelta²):

| Variante | Sesgo | MAE | MAE ponderado |
|---|---|---|---|
| legacy | −0.0115 | 0.0363 | 0.0227 |
| ninguno | −0.0122 | 0.0366 | 0.0232 |
| conductual 0.84 | −0.0100 | 0.0356 | 0.0217 |
| físico 0.66 | −0.0078 | 0.0347 | 0.0205 |
| físico 0.555 | −0.0068 | 0.0343 | 0.0199 |

En los 8 GP del paper, el orden es el mismo (MAE 0.0359 → 0.0334). La forma de la degradación dentro de cada stint de
2024 casi no distingue entre variantes (RMSE 0.61–0.67 s, dominado por ruido).

Conclusiones:
- Por vuelta, el neumático de 2021 se degradaba en tiempo de vuelta a algo más de la mitad del ritmo de 2022-23.
- En cambio, los stints de 2021 eran solo un 19 % más largos (factor conductual 0.84). Los dos factores no son
  compatibles entre sí: la decisión de parar no seguía solo a la degradación.
- Para los modelos de tiempo de vuelta, el factor físico mejora la predicción de las pendientes de 2024. Para el logit
  de paradas de los rivales, un factor más fuerte empeora levemente 2024 (sección anterior).
- 2022 y 2023 son homogéneos, así que la referencia está bien definida.
- En la Ec. 1 por GP, el cambio solo afecta a los GP con datos de 2021 (de los 8 del paper: Bahrain, Dutch, Hungarian,
  Saudi y US; no Belgian, Miami ni Singapore). En Saudi, las pendientes negativas pasan a ser cercanas a 0. El GLS
  propaga el cambio a todos los GP.

## 2026-10-15 (tarde): factor conductual con el último stint
Criterio del autor: el último stint también informa cuánto aguanta el neumático, porque las paradas se planifican sabiendo
cuántas vueltas quedan. `stint_factors_v2_20261015.py` (log en `stint_factors_v2_20261015.log`) agrega los últimos stints
de pilotos que terminan, que empezaron con una parada en verde y tienen ≥ 8 vueltas (≥ 12 como sensibilidad).
- 2021: f = 0.859 (IC 0.79–0.94), frente a 0.841 sin los últimos stints. Con control por último stint 0.871; con ≥ 12
  vueltas 0.852. Por compuesto: C2 .86, C3 .88, C4 .77, C5 .94 (C0 .57, una sola celda).
- 2024 frente a 2022-23: 0.96 (0.90–1.03), sin diferencia.
- Conclusión: el factor conductual queda en ≈ 0.86 y sigue lejos del físico (0.555).

Decisiones del autor (15-oct): el logit de adelantamiento se reestima en v7 (usa `TyreLife` 2021 directamente y los
tiempos predichos de la Ec. 1; sus tablas del paper cambiarán). El logit condicional (K1) entra en v7.

## 2026-10-15: 0.555 frente a 0.66 (curva de combustible por año)
`fuel_by_year_20261015.py`: la diferencia es que 0.66 deja que la tendencia por número de vuelta (combustible y
evolución de pista) sea distinta por año. Esa tendencia difiere claramente entre años: χ² ponderado ≈ 1845 con 54 gl,
sobrestimado porque los residuos de una carrera están correlacionados, pero muy grande igual. Mejora mediana por
vuelta (vueltas 1–40): 2021 −0.069 s, 2022 −0.049 s, 2023 −0.048 s. Con una curva común, parte de la mayor mejora
de 2021 se atribuye a menor degradación, y el factor baja a 0.555. Por compuesto: un factor común basta (p = 0.17
en tiempos de vuelta, p = 0.27 en paradas sin C0), y en la validación 2024 da el mismo error que uno por compuesto.

## 2026-10-15: factor dentro del logit de paradas (análogo del 0.66)
`stop_logit_factor_20261015.py`: perfil de verosimilitud de f en el logit de parada (TyreLife 2021 × f), siempre con
efecto propio de 2021. L1 (S0 + y21) 0.864 (0.79–0.94); L2 (tramo de carrera flexible) 0.877 (0.81–0.95);
L3 (curva de tramo de carrera propia de 2021, análogo de la curva de combustible por año) 0.785 (0.71–0.87), con
mejora significativa sobre L2 (LR ≈ 29.6, 5 gl); L4 (L3 + LapsLeft y SC/VSC propios de 2021) 0.792 (0.71–0.88).
IC sin agrupar por carrera (optimistas). Con la misma lógica, el conductual baja a ≈ 0.79 y el físico sube a 0.66:
la brecha se achica y los IC se tocan (0.52–0.80 y 0.71–0.88).

## 2026-10-15: factor único (decisión del autor), estimación conjunta
`joint_factor_20261015.py`: promedio ponderado por precisión de la estimación física (0.660, ee 0.072) y la conductual
L3 (0.786, ee 0.080; ee recalculado agrupando por carrera, antes el IC del perfil era optimista). Resultado:
**f = 0.716 (ee 0.053, IC 0.61–0.82)**, compatibles (p = 0.24). Con L4: 0.712. Con las especificaciones sin curva
propia de 2021 (0.555 y L2) son incompatibles (p = 0.011), lo que confirma que la brecha venía de ahí.
Guardado en `data/tyrelife_2021_factors_v7.json` (`F1_TYRELIFE_2021=v7`), igual para todos los compuestos.
