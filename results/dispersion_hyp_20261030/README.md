# Dispersión de las posiciones finales en v10eGQf: hipótesis y pruebas (30-oct-2026)

Contexto: en la Fig. 7, la de Hungarian casi no tiene dispersión, y la cobertura del 90 % con la estrategia real
bajó de 89 % (v9r2) a 80 % (v10f). La ablación de Hungarian está en
`results/real_strategy_check_v10eGQf_20261029/README.md`: la dispersión colapsa al combinar el logit en todo gap con
el gap medio. Todo se corrió en el Mac, sin el cluster y sin reentrenar.

Carpetas:
- `results/overtake_noise_frac_20261030/`: variantes, corridas y comparaciones.
  - `run_variant_20261030.sh <tag> <count|fig7|fig6> [GP] [n]` corre una variante;
  - `run_fig7_20261030.sh` corre las 8 GP;
  - `ot_count_20261030.py` + `summ_count_20261030.py` dan los adelantamientos por tramo y los gaps
    → `count_summary_20261030.txt`;
  - `compare_lambda_20261030.py` da la cobertura → `compare_lambda*_20261030.txt`, `compare_fig6_20261030.txt`.
- Esta carpeta:
  - `race_effect_20261030.py`: estimación del efecto piloto-carrera;
  - `blocked_gaps_20261030.csv`: gaps reales de los autos bloqueados;
  - `world2024_r24/`: los agentes actuales en el mundo con efecto por carrera.

## Opciones nuevas del entorno

Todas están en `src/environment/f1_env_all_drivers.py` y son opt-in. Por defecto no cambia nada: con las opciones
apagadas, los adelantamientos simulados son idénticos (`l000_chk`) y la validación con la estrategia real (`l000`) da lo
mismo que la vigente.

| Variable | Qué hace |
|---|---|
| `F1_OVERTAKE_NOISE_FRAC=λ` | Con el gap medio, el logit se evalúa en el gap medio + λ × la diferencia de los ruidos de la vuelta. λ = 0 es v10f; λ = 1 es el gap con ruido. |
| `F1_RACE_EFFECT_SD=σ` | Al inicio de la carrera, cada piloto sortea u ~ N(0, σ), que se suma a todos sus tiempos de vuelta en pista libre (medio y con ruido). El ruido iid se reduce a √(sd² − σ²), así que la varianza por vuelta no cambia. |
| `F1_RACE_EFFECT_LOGIT=0` | El logit se evalúa sin u. Por defecto u entra en el ritmo medio. |
| `F1_FOLLOWING_GAP_EMP=<csv>` | El gap de seguimiento se remuestrea de los gaps reales de los autos bloqueados del GP. Con `F1_FOLLOWING_GAP_EMP_POOLED=1` se usan los de todos los GP. |

Tags de `run_variant`: `lXYZ` = λ 0.XYZ; `E` = sin la opción del gap medio; `f1` = seguimiento Exp(1); `rNN` = σ 0.NN;
`x` = u fuera del logit; `emp` / `empP` = gap empírico del GP / de todos los GP.

## 1. Efecto piloto-carrera en los residuos de la Ec. 1 [V]

`race_effect_20261030.py`: Ec. 1 de cada GP (2021–23, como el árbol v10eGQ, sin los overrides de neumático).

- **Residuos:** sd ≈ 0,55–0,79 s. La media del residuo por piloto-carrera tiene una sd de 0,19–0,38 s/vuelta, contra
  0,08–0,12 que se esperaría si los residuos fueran iid. La autocorrelación de orden 1 es de 0,39–0,55.
- **Efecto propio de la carrera,** neto de la media del piloto en la temporada sobre los 8 GP (n = 350):
  - sd = **0,235 s/vuelta**, IC 95 % bootstrap [0,196; 0,277];
  - 0,24 / 0,23 / 0,24 en 2021 / 2022 / 2023;
  - versión robusta (MAD) ≈ 0,18.
- **Interpretación:** en una carrera, un piloto rinde de forma persistente más o menos que lo que predice la Ec. 1, por
  ejemplo por el auto, el ajuste o la forma de ese día. En 60 vueltas son unos 14 s, contra ~5 s del ruido iid.
- **Lo que hace hoy el simulador:** ruido iid por vuelta, así que esa variabilidad no existe. Con el gap medio y el
  bloqueo, además, el ruido iid casi no mueve posiciones.

## 2. Resultados (estrategia real, Fig. 7, 8 GP × 500 carreras; adelantamientos: 100 carreras por GP) [V]

Cobertura: pilotos que terminaron en 2024 y en ≥ 95 % de las simulaciones (n = 152). Las columnas de adelantamientos son
la suma de los 8 GP de la media por carrera, en vueltas 5+ verdes, en tramos de gap ≤ 1 / 1–2 / > 2 s.

| Variante | Cob. 90 % | Ancho 90 % | Err. abs. mediana | Spearman | PIT por quintil | Adel. ≤1 / 1–2 / >2 | % pares ≤ 1 s |
|---|---|---|---|---|---|---|---|
| Real | 90 % (objetivo) | | | | uniforme | 289 / 68 / 114 (470) | 19,6 |
| v10f vigente (`l000`) | 80,3 % | 3,9 | 1,66 | 0,907 | 33 28 17 23 51 | 341 / 102 / 122 (564) | 27,1 |
| λ = 0,5 | 81,6 % | 4,1 | 1,65 | 0,906 | | 360 / 118 / 123 (601) | 26,1 |
| λ = 1 | 82,2 % | 4,5 | 1,74 | 0,903 | | 422 / 157 / 134 (713) | 25,2 |
| E (sin gap medio) | 82,2 % | 4,8 | 1,76 | 0,895 | | 412 / 161 / 135 (709) | 24,9 |
| λ = 1, seguimiento Exp(1) | 82,9 % | 4,6 | 1,55 | 0,917 | | 167 / 172 / 177 (515) | 10,1 |
| E, seguimiento Exp(1) | 87,5 % | 6,5 | 1,72 | 0,906 | | 157 / 172 / 182 (512) | 9,4 |
| gap de seguimiento empírico del GP | 77,6 % | 3,9 | 1,64 | 0,910 | | 316 / 111 / 120 (547) | 25,4 |
| gap de seguimiento empírico, todos los GP | 80,9 % | 4,0 | 1,65 | 0,910 | | 312 / 109 / 120 (541) | 25,2 |
| **σ = 0,24 (`l000_r24`)** | **90,1 %** | 5,9 | **1,54** | **0,912** | **29 26 32 32 33** | 347 / 104 / 119 (569) | 27,3 |
| σ = 0,15 | 85,5 % | 4,8 | 1,64 | 0,910 | | 344 / 101 / 118 (564) | 27,4 |
| σ = 0,24, u fuera del logit | 83,6 % | 4,5 | 1,63 | 0,909 | | 346 / 100 / 117 (563) | 28,6 |
| σ = 0,24 + gap empírico | 89,5 % | 5,8 | 1,57 | 0,912 | | 321 / 114 / 120 (555) | 25,2 |
| σ = 0,24, λ = 0,5 | 91,4 % | 6,1 | 1,60 | 0,908 | | 367 / 117 / 125 (609) | 26,0 |
| σ = 0,24, λ = 1 | 92,1 % | 6,6 | 1,59 | 0,907 | | 420 / 153 / 134 (707) | 24,8 |

Las sumas de adelantamientos de este entorno (el de la validación, con los juegos de neumáticos, etc.) son mayores que
las del README de v10f (564 contra 491 de `tree:W`). Por eso las variantes se comparan con `l000`, no con esa cifra.

La Fig. 6 (rivales estocásticos) ya cubría 92,1 %. Con σ = 0,24 sube a 95,4 % (σ = 0,15: 93,4 %).

Hungarian TSU con la estrategia real:

| Variante | Media ± de | Intervalo del 90 % |
|---|---|---|
| Vigente | 8,14 ± 0,84 | [7, 9] |
| λ = 1 | 8,88 ± 1,56 | |
| σ = 0,24 | 8,60 ± 1,65 | [7, 11] |
| Real | 9 | |

Miami pasa de 63 % a 89 % de cobertura y US de 63 % a 84 %.

## 3. Lectura

1. **λ = 1 (ruido del tiempo de vuelta en el logit):** casi no mejora la cobertura (80 → 82 %) y sube los adelantamientos
   un 26 %, sobre todo a 1–2 s (157 contra 68 reales). El ruido iid por vuelta no es lo que falta: se promedia a lo
   largo de la carrera. Que la etiqueta del logit ya contiene ese ruido está verificado en el README de v10f.
2. **Gap de seguimiento:** el de 0,4 + Exp(0,26) ya es aleatorio, se sortea por par y por vuelta. En los datos,
   Hungarian no sigue más cerca que los demás: la mediana de los bloqueados es 0,73 s, contra 0,62–0,63 en Dutch, Miami
   y Saudi. Remuestrear los gaps reales del GP no cambia la cobertura (77,6 %). Exp(1) da más dispersión, pero separa
   los autos: solo 9–10 % de los pares quedan a ≤ 1 s, contra 19,6 % real.
3. **Efecto piloto-carrera:** con σ = 0,24, estimado de los datos y no ajustado a la cobertura, la cobertura queda en el
   nominal (90,1 %). El PIT queda casi uniforme, y la mediana y el Spearman mejoran. Los adelantamientos y la
   distribución de gaps no cambian.
   - u tiene que entrar al logit: sin eso, el auto más rápido queda bloqueado y la cobertura vuelve a 84 %.
   - Ojo: con el mismo argumento de la doble cuenta, u ya está en la etiqueta del logit. Pero con u los adelantamientos
     no suben (569 contra 564), así que en la práctica no hay doble cuenta medible.

Recomendación: v10f + `F1_RACE_EFFECT_SD=0.24`, con u en el logit y λ = 0.

## 4. Agentes actuales en el mundo σ = 0,24 (sin reentrenar) [V]

`world2024_r24/`: pipeline v10 (etapas fig6, fig7, summary, figs) con `F1_RACE_EFFECT_SD=0.24`, agentes
`retrained_v10eGQf_20261029/selected` (son los de antes del reentrenamiento con ent 0,05 de PPO), 500 carreras. La caja
Actual de la Fig. 7 sale de `l000_r24`. Ver `world2024_r24/fig{6_2024_realdnf,7_2024}/figure_*_2x4*.png`.
Comparación: `compare_agents_20261030.py` → `compare_agents_20261030.txt`.

Diferencia con la QMIP (negativo = mejor), vigente → r24:

| | DQN | A2C | TRPO | PPO | RPPO |
|---|---|---|---|---|---|
| Fig. 6, 8 GP | −0,13 → −0,07 | +0,15 → +0,20 | +0,56 → +0,64 | +0,09 → +0,03 | +0,02 → +0,01 |
| Fig. 6, sin Miami | −0,15 → −0,09 | +0,03 → +0,10 | +0,49 → +0,57 | −0,26 → −0,27 | −0,34 → −0,29 |
| Fig. 7, 8 GP | +0,10 → +0,08 | −0,27 → −0,31 | +0,36 → +0,38 | +0,48 → +0,30 | +0,42 → +0,24 |
| Fig. 7, sin Miami | +0,25 → +0,18 | −0,31 → −0,34 | +0,24 → +0,26 | −0,23 → −0,29 | −0,30 → −0,36 |

(IC 95 % ≈ ± 0,06–0,10.) El orden entre métodos y los signos no cambian. Lo que cambia es la dispersión:

- **Fig. 7, de de la posición:** la mayoría de las cajas se ensancha entre 30 y 100 %.
- **Hungarian TSU:** de 0,8–0,9 a 1,6–1,75; QMIP de 1,37 a 2,53.
- **Hungarian, carreras que terminan P8:** de 58–62 % a 30–36 % (QMIP: de 42 % a 24 %).
- **Por qué sigue la moda en P8:** es en parte estructural. En 2024, TSU corrió como "mejor del resto", detrás de los
  8 autos de los 4 equipos grandes: en la realidad estuvo 27 de 70 vueltas en P9 y terminó P9, a una mediana de 5,2 s
  del auto de adelante.

Los agentes se entrenaron sin u. Para el paper habría que reentrenar en el mundo con u, para que entrenamiento y
evaluación coincidan. La evidencia de acá (sin reentrenar, conclusiones iguales) sugiere que el reentrenamiento no
debería cambiar la historia, pero eso es una hipótesis.

## 5. Si se adopta (pendiente de decisión del autor)

1. Exportar `F1_RACE_EFFECT_SD=0.24` en el envoltorio v10f (o en uno nuevo) y en `pipeline_v10_20261027.sh`.
2. Reentrenar multi (400 tareas) + PPO/RPPO ent 0,05 en el cluster, y rehacer la selección, las Figs. 6–7, la Tabla
   O.1 y la validación con la estrategia real.
   - El mono-piloto no tiene rivales: u solo mueve el tiempo total. No es necesario reentrenarlo, por decidir.
3. Paper: una frase en la Ec. 1 / Alg. 2 (efecto aleatorio piloto-carrera N(0, 0,24²), con el ruido iid reducido para
   mantener la varianza por vuelta), y el ítem correspondiente del suplementario con la estimación (§1).

## 6. Relación con el error de la Ec. 2 (30-oct) [V]

El simulador usa el coeficiente de piloto predicho por la Ec. 2 como valor fijo, sin ruido; antes de `F1_RACE_EFFECT_SD`
no había ningún término persistente por piloto-carrera. Error de la Ec. 2 porcentual
(`results/driver_coef_v10_20261028/v10eGQ/eq2_predictions.csv` y `race_coefs.csv`), sin Australia ni VER, descontado el
desplazamiento común de cada carrera:

| Muestra | n | sd del error | Neto del error de muestreo (iid) | Neto (AR 0,5) |
|---|---|---|---|---|
| 2023, 8 GP | 82 | 0,267 | 0,248 | 0,207 |
| 2024, 8 GP | 126 | 0,301 | 0,284 | 0,246 |
| 2024, todos | 333 | 0,329 | 0,311 | 0,273 |

El ruido de la Ec. 2 y el efecto piloto-carrera son el mismo fenómeno, medido por dos vías, y dan magnitudes compatibles
(0,21–0,28 frente a 0,235). No hay que sumarlos: σ = 0,24 ya representa la incertidumbre de la Ec. 2.

## 7. Ruido de la Ec. 2 en vez del efecto piloto-carrera (30-oct, propuesta del autor) [V]

Opción nueva `F1_RACE_EFFECT_KEEP_STD=1` (tag `k`): el mismo sorteo por piloto y carrera, leído como error de predicción del
coeficiente de la Ec. 2; el ruido por vuelta de la Ec. 1 no se achica. σ = 0,28 (neto) y 0,30 (bruto), de la tabla del §6.
Fig. 7, 8 GP × 500 carreras (`../overtake_noise_frac_20261030/compare_eq2noise_20261030.txt`):

| Variante | Cobertura 90 % | Ancho | Quintiles PIT | MAE mediana | Spearman | Adelantamientos (real 470) |
|---|---|---|---|---|---|---|
| vigente (l000) | 80,3 % | 3,9 | 37 19 24 22 50 | 1,66 | 0,907 | 564 |
| r24 (efecto, ruido achicado) | 90,1 % | 5,9 | 29 26 32 32 33 | 1,54 | 0,912 | 569 |
| **r28_k (Ec. 2, 0,28)** | **92,8 %** | 6,4 | 27 28 34 33 30 | 1,53 | 0,912 | 573 |
| r30_k (Ec. 2, 0,30) | 92,8 % | 6,7 | 23 28 36 36 29 | 1,55 | 0,912 | — |

Tareas preparadas (no lanzadas) para el piloto de Hungría: `scripts/eq2_noise_20261030.py` (σ 0,28) y
`reproduce/cluster/tasks_v10eGQfq_hun_20261030.tsv` (50 tareas, PPO con ent 0,05; salida `retrained_v10eGQfq_20261030`).

## 8. v10g: piloto PPO en los 8 GP lanzado (30-oct, OK del autor)

- **v10g** = v10eGQf + ruido de la Ec. 2 (σ 0,28, `scripts/eq2_noise_20261030.py`). Tareas:
  `reproduce/cluster/tasks_v10g_ppo8_20261030.tsv` (80 = PPO ent 0,05 × 8 GP × 10 semillas; `make_tasks_v10g_20261030.py`,
  que también genera `tasks_v10g_hun_20261030.tsv`, 50 tareas de Hungría con los 5 métodos, no lanzadas).
- Salida en el cluster: `trained_models/retrained_v10g_20261030/`.
- Subido: `src/environment/f1_env_all_drivers.py` (antes = versión previa a las opciones opt-in, md5 fe541b79) y el
  envoltorio. rsync por el alias `nlhpc` (la IP no respondió el 30-oct).
- Prueba: 18138628 (tareas 22 = Miami s1 y 23 = Hungría s1), 830M. Logs con `[eq2 noise]`, `[v10f]`, `ent_coef=0.05`,
  ~1000 fps, 0,40 GB a los 10 min.
- Arreglo: 18138645 (tareas 1-21 y 24-80, `%20`), 830M.
- Después: bajar sin `--delete`, seleccionar semilla y correr Figs. 6–7 de PPO vs QMIP en v10g, comparando con los PPO
  actuales (v10f, ent 0,05).
