# Variabilidad del tiempo de parada: ¿falta ruido en el simulador? (2026-10-31)

Análisis exploratorio pedido por el autor antes de lanzar el reentrenamiento de v10g, para decidir si
incorporar una varianza propia del pit stop. **Decisión del autor: NO incorporarlo.** Se deja documentado
porque el hallazgo sobre el filtro de outliers es de interés propio.

Datos: `data/session_{2021,2022,2023,2024}_V2.csv`. Una "parada" = in-lap (`PitIn=1`) y la out-lap
siguiente (`PitOut=1`), ambas con `TrackStatus=1`. La pérdida se mide contra la mediana de las vueltas
limpias vecinas (ventana ±5 del mismo piloto, sin pit laps); el residuo es dentro de cada GP.
Reproducir, desde la raíz del repo y en este orden:

    python results/pitstop_var_20261031/pit_loss_20261031.py      # escribe pit_loss.csv
    python results/pitstop_var_20261031/stop_level_20261031.py
    python results/pitstop_var_20261031/tail_triage_20261031.py
    PYTHONPATH=src:. python results/pitstop_var_20261031/iqr_survival_20261031.py
    python results/pitstop_var_20261031/sec_per_pos_20261031.py
    python results/pitstop_var_20261031/impact_20261031.py

`iqr_survival_20261031.py` llama a `filtrar_outliers_iqr` de `src/models/regressions.py`, que no escribe
nada (a diferencia de `scripts/train_regressions.py`, que no debe correrse).

## 1. La distribución es bimodal [V]

Pérdida total de una parada: 18,3 a 30,2 s según el circuito. El residuo dentro de cada GP (n = 2.100):

| | n | % | dispersión / exceso |
|---|---|---|---|
| Núcleo (parada normal) | 1.833 | 87 % | sd 1,82 s; IQR/1,349 = **1,62 s** |
| Cola (exceso > 5 s) | 197 | **9,4 %** | exceso medio **12,2 s**; mediana 9,0 s; p90 24,1 s |

sd cruda 5,6 s contra sd robusta 2,1 s: el factor 2,7 es la firma de la cola. De la cola, solo un 11 %
tiene otra parada a ≤3 vueltas (reparación o doble) y un 6 % está a ≤3 vueltas del fin de su carrera
(abandono), así que es mayoritariamente genuina. **Limitación:** no se puede separar una parada
mecánicamente mala de una penalización de 5 o 10 s cumplida en boxes, que no es ruido operativo.

## 2. El filtro de outliers elimina la cola, y solo de las pit laps [V]

El filtro `median + 15 s` de `all_laps()` **no** toca las paradas: tiene una excepción explícita para
`PitIn`, `PitOut` y vuelta 1 (`src/models/regressions.py:95`). El que las elimina es el IQR iterativo
k = 3 sobre los residuos (`filtrar_outliers_iqr`), de forma muy desproporcionada:

| GP | in-laps eliminadas | out-laps eliminadas | todas las vueltas |
|---|---|---|---|
| Bahrain | 3 % | 12 % | 0,8 % |
| Saudi | 17 % | 35 % | 1,2 % |
| Miami | 6 % | 24 % | 1,1 % |
| Hungarian | 2 % | **48 %** | 2,9 % |
| Belgian | 5 % | 7 % | 1,9 % |
| Dutch | **35 %** | 26 % | 3,3 % |
| Singapore | 20 % | **67 %** (de 9) | 3,6 % |
| United States | 8 % | 13 % | 1,7 % |

Es decir, β'_PitIn y β'_PitOut quedan estimados sobre paradas limpias y el σ del modelo (0,55 a 1,01 s
por GP) es el de vueltas normales.

## 3. Cuánto faltaría [E]

El simulador ya sortea ruido iid en las dos vueltas de la parada: ≈ 0,8 a 1,4 s de sd en la pérdida
total, contra 1,62 a 1,82 s del núcleo real. **El núcleo ya está esencialmente cubierto; lo que falta es
la cola.** Inyectando la distribución empírica del exceso sobre los resultados de la Fig. 7 y traduciendo
segundos a posiciones con los gaps reales de 2024 (20.000 réplicas por GP):

| GP | sd posición actual | con ruido de parada | Δ media | P(perder ≥2 pos.) |
|---|---|---|---|---|
| Bahrain | 2,45 | 2,53 | +0,22 | 4,9 % |
| Saudi | 1,58 | 1,82 | +0,29 | 6,4 % |
| Miami | 3,75 | 4,07 | +0,64 | 13,7 % |
| Hungarian | 1,75 | 1,94 | +0,27 | 6,1 % |
| Belgian | 2,87 | 3,15 | +0,48 | 10,6 % |
| Dutch | 2,33 | 2,48 | +0,28 | 6,5 % |
| Singapore | 1,90 | 2,02 | +0,24 | 5,5 % |
| United States | 2,22 | 2,36 | +0,23 | 5,3 % |
| **Media** | **2,36** | **2,55** | **+0,33** | **7,4 %** |

Los peores son Miami y Bélgica, que son los circuitos con el pelotón más apretado (gap mediano 2,3 y
2,8 s contra 5 a 6 s en el resto). En ninguno la sd sube más de un 10 %. La traducción a posiciones es
aproximada [E]: ignora la recuperación posterior (sobreestima) y el riesgo de quedar atrapado en tráfico
(subestima); léase como orden de magnitud.

## 4. Sería simétrico entre PPO y QMIP [V]

En los CSV de la Fig. 7 de v10g, PPO y QMIP hacen exactamente 1,00 paradas en los 8 GP, así que esto
ensancharía las dos cajas por igual sin cambiar la comparación del paper.

## 5. Conclusión

Efecto de segundo orden frente al ruido de la Ec. 2 ya incorporado (que movió Hungría de ±0,8 a ±1,75).
Si alguna vez se retoma, lo natural es tratarlo como ruido de evaluación y no meterlo en el mundo de
entrenamiento: si el agente aprende que parar puede costar 12 s extra con 9 % de probabilidad, cambia su
política para parar menos, y eso sí alteraría la comparación con la QMIP, que optimiza sin ese riesgo.
