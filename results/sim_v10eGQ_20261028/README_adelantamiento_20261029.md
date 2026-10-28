# Adelantamiento v10f (28-29 oct 2026)

Cambios al modelo de adelantamiento del simulador multi-piloto v10eGQ, aprobados por el autor el 29-oct. El simulador
mono-piloto y la QMIP no usan adelantamientos y no cambian.

## Qué cambió

| | Antes (v10eGQ, 28-oct) | Ahora (v10f) |
|---|---|---|
| Etiqueta | gana posición neta y TrackStatus(L) == 1 | por par: termina la vuelta L+1 delante del auto que tenía adelante |
| Muestra | todas las vueltas (con SC/VSC en L+1 la etiqueta era 0) | vuelta L+1 sin SC, VSC ni roja |
| Dónde se aplica el logit | pares con gap inicial ≤ 1.0 s (fuera: adelanta ⇔ gap predicho < 0) | todos los pares en pista libre (`F1_OVERTAKE_MAX_GAP=inf`) |
| Gap con que se evalúa | gap inicial + diferencia de tiempos con ruido | gap inicial + diferencia de tiempos **medios** (`F1_OVERTAKE_MEAN_GAP=1`) |
| Gap de seguimiento tras no adelantar | 0.4 + Exp(1) s | 0.4 + Exp(0.26) s |

Archivos:
- estimación: `overtake_v10f_20261029.py` (en la cadena `run_chain_v10e_20261028.sh`, reemplaza a `overtake_v10_20261027.py`);
  salida en `overtake_v10f/` y `artifacts/<GP>/overtake_{logit,threshold}.pkl`;
- simulador: `src/environment/f1_env_all_drivers.py` (opciones `overtake_max_gap`, `overtake_mean_gap`, campo
  `Driver.mean_laptime`; por defecto se mantiene el comportamiento anterior, verificado bit a bit en 15 carreras);
- envoltorio `scripts/overtake_v10f_20261029.py`, que fija las tres opciones y fuerza `--following-gap-scale 0.26`;
- árboles `trained_models/simulation{,_onedriver}_v10eGQ_20261028`, reconstruidos con el mismo nombre (solo cambian
  `overtake_logit.pkl` y `overtake_threshold.pkl` del multi-piloto); los anteriores están en `*_old_preadelantamiento`;
- tareas `reproduce/cluster/tasks_v10eGQf_20261029.tsv` (`scripts/make_tasks_v10eGQf_20261029.py`), con salida en
  `trained_models/retrained_v10eGQf_20261029/`. Las del mono-piloto siguen siendo `tasks_onedriver_v10eGQ_20261028.tsv`.

Los parámetros (const, pendiente por GP) están en `overtake_v10f/params.csv`. Las pendientes van de −1.5 (Saudi Arabia) a
−2.6 (Bélgica); antes las curvas de Saudi Arabia y Singapur eran casi planas.

## Diagnóstico (scripts y salidas en `overtake_diag_20261029/`)

1. **Curvas planas (Saudi Arabia, Singapur) y contradicciones (Bélgica, Bahrein).** Entre 11 % y 26 % de la muestra eran
   vueltas no verdes con etiqueta forzada a 0. Además, la etiqueta neta no contaba 46 adelantamientos a un rival que
   para en boxes. Con la vuelta L+1 sin SC/VSC y la etiqueta por par, las pendientes quedan entre −1.5 y −2.6 y el ajuste
   coincide con las tasas por tramo (`ot_variants.py`, `ot_pair.py`, `ot_support.py`).
2. **Umbral de 1 s y ruido (`ot_gap.py` → `umbral_1s_datos.txt`).** La etiqueta coincide al 100 % con "llega antes con
   los tiempos reales", así que el logit ya contiene la incertidumbre del tiempo de vuelta. Evaluarlo sobre el gap con
   ruido (σ ≈ 0.7 s por piloto) la cuenta dos veces. Dentro de 1 s se esperaban 942 adelantamientos frente a 758 reales;
   con el gap medio, 736. Fuera de 1 s, la regla determinista con ruido da 447 frente a 178 reales entre 1 y 2 s,
   mientras que el logit da 222.
3. **Gap de seguimiento (`persist.py`, `blocked.py` → `persistencia_gaps_reales.txt`).**
   - El simulador dejaba a todo par que no adelanta a 0.4 + Exp(1), con media 1.4 s, en cada vuelta. Por eso solo el
     8.8 % de los pares adyacentes empezaba la vuelta a ≤ 1 s, frente a 19.6 % en la realidad.
   - En los datos, el 79 % de los pares a 0.5-1 s que no se adelantan sigue a ≤ 1 s. Los autos bloqueados (gap inicial
     ≤ 1 s, más rápidos en la predicción, sin adelantar) quedan a mediana 0.62 s y media 0.66 s, lo que da Exp(0.26).
   - Un gap constante de 0.66 s pega a los autos (30.5 % ≤ 1 s, 10 % de los pares exactamente a 0.66 s).
     Remuestrear los gaps reales da lo mismo que 0.4 + Exp(0.26).
4. **Simulador (`ot_sim_test.py`, `ot_sim_phys.py`, `phys_cmp.py` → `sim_variantes.txt`, `sim_por_gp.txt`).**
   100 carreras por GP. El agente sigue una estrategia fija de una parada y se excluyen sus pares. Se cuentan
   adelantamientos en vueltas 5+ sin SC/VSC; la suma de los 8 GP es la media por carrera.

   | Variante | % pares ≤ 1 s | Mediana gap | Adel. ≤1 / 1-2 / >2 s | Total |
   |---|---:|---:|---|---:|
   | Real (21 carreras) | 19.6 | 2.69 | 289 / 68 / 114 | 470 |
   | Antes | 8.8 | 3.30 | 74 / 239 / 214 | 526 |
   | Solo logit nuevo sin umbral + gap medio | 10.3 | 3.34 | 120 / 108 / 138 | 366 |
   | **v10f (árbol nuevo + envoltorio)** | 26.5 | 2.37 | 295 / 88 / 109 | 491 |

   - El ruido de los tiempos de vuelta no explica la dispersión de gaps: el entorno determinista da 9.0 %.
   - Por GP, el error absoluto medio en adelantamientos por carrera baja de 17.4 a 13.2. Bélgica y Miami siguen por debajo
     de lo real. Solo hay 2-3 carreras reales por GP.
5. **Forma funcional (`ff.py` → `formas_funcionales.txt`, `calibracion_lineal_asinh.txt`).**
   - Se probaron cuadrática, cúbica, raíz, asinh y pendiente distinta a cada lado de 0, con el término adicional por GP o
     común, y se compararon con AIC, BIC y validación cruzada dejando una carrera fuera.
   - La cuadrática gana en validación cruzada, pero no es monótona: en 6 GP la probabilidad vuelve a subir desde
     δ ≈ +4-5 s, y con δ = +10 s da 47-99.8 %.
   - asinh(δ) es monótona y mejora poco (ΔBIC −52, +26 de log-verosimilitud en validación cruzada). Su calibración es
     peor en (−5, −2] s, donde falla el 4.5 % de los adelantamientos frente a 0.6 % real.
   - Se mantiene la forma lineal (decisión del autor).

Verificación:
- Con las opciones apagadas, el entorno nuevo es igual bit a bit al anterior (`equiv.py`).
- Con el envoltorio, el árbol nuevo reproduce la variante probada (491 frente a 491).
- Prueba corta de entrenamiento con PPO y DQN, Singapur, 4000 pasos: sin errores y a ~1700 pasos/s, como antes.

Una primera versión de `overtake_mean_gap` restaba el ruido del gap ya ajustado por los adelantamientos previos de la
misma vuelta. Eso inflaba los adelantamientos en cadena tras una parada (577 en total). Se corrigió a la definición de la
estimación: gap inicial más la diferencia de medias.

## Para el manuscrito (por proponer en `\vten{}`)

- Ec. 3 y su texto: la muestra (vuelta siguiente sin SC/VSC/roja), la etiqueta por par y que se usan todos los gaps.
- Algoritmo 2 y el suplementario:
  - el logit se aplica a todo par en pista libre con el gap predicho medio;
  - el gap de seguimiento es 0.4 + Exp(0.26), calibrado con los autos bloqueados;
  - se elimina la frase "reproduces the dispersion of real intervals".
- Tabla de coeficientes de adelantamiento, si está en el suplementario.
