# Edad efectiva de los neumáticos: prueba preliminar de la Ec. 1 (28-oct-2026)

Pedido del autor: que no cuenten como vuelta de uso (i) las vueltas bajo SC o VSC y (ii) las vueltas lentas de prácticas
y clasificación. Hoy los datos (FastF1) cuentan todas las vueltas; el simulador multi-piloto congela la edad bajo SC/VSC.

1. `extract_fpq_20261028.py` (entorno `~/.venvs/f1-fastf1`, caché local offline) → `fpq_laps_2021_2024.csv.gz`:
   vueltas de prácticas y clasificación 2021–2024. Las sesiones sprint no están en la caché (y la API de F1 responde
   401), así que no quedaron; se agregan en el paso 1b.
1b. `extract_sprint_20261028.py` → `fpq_sprint_laps_2021_2024.csv.gz` (log `extract_sprint.log`), agregado el 28-oct:
   - 2021 y 2022, sprint: FastF1 contra el mirror de live timing (en 2021 la carpeta del feed es `*_Sprint_Qualifying/`);
   - 2023, shootout y sprint, y 2024, sprint qualifying: OpenF1 (sin TrackStatus por vuelta; la vuelta de entrada se
     toma como la última de cada tanda; se omiten 11 tandas sin vueltas o edad, 10 de ellas en la sprint de Bélgica 2023);
   - 2024, sprint: `data/practice_laps_2024_20260930.csv`.
2. `ages_20261028.py` → `ages_2021_2024.csv.gz`, `stint_starts.csv`, `fpq_stints.csv`. Variantes de edad:
   - `G`: sin vueltas de carrera con SC, VSC o bandera roja;
   - `Q`: la edad con que llega un juego usado cuenta solo vueltas de empuje (≤ 1,10 × la mejor del piloto en la
     sesión, sin entrada, salida ni bandera), siguiendo la cadena de tandas del juego en prácticas y clasificación;
   - `GQ`: ambas.
   Las sesiones se ordenan por su cronología real (en los fines de semana sprint el orden cambia según el año).
3. `compare_20261028.py`: la Ec. 1 por GP, con la misma muestra y fórmula que hoy (datos v10d, filtro A, factor 2021
   de 0.74), solo cambiando `TyreLife`.
4. `grid_20261028.py`: grilla de pesos `TyreLife = GQ + θ·(vueltas SC/VSC) + φ·(vueltas lentas previas)`.
5. `grid_quali_20261028.py` → `grid_quali.csv`, `grid_quali_summary.txt`: peso ψ de las vueltas de empuje de
   clasificación (Q, SQ y shootout) en la edad con que llega un juego usado; ψ = 1 es `GQ`.

## Resultados [V]

- **Juegos usados:** 20 % de las tandas de carrera (982 de 4 808). Se rastrea el 87,7 % de sus vueltas previas: el
  94,8 % en los juegos de largada y el 83,2 % en los puestos en una parada. El resto se completa con la proporción
  global de vueltas de empuje (0,268). La edad media con que llega un juego usado baja de 2,9 a 0,8 vueltas. El 97 % de
  las vueltas de empuje rastreadas son de clasificación.
- **Corrección del 28-oct (después de la cadena v10e):** con pandas 3 dos compuestos `NaN` no son iguales, y cada vuelta
  sin compuesto (Austria y Mónaco 2022) se tomaba como una tanda nueva con juego usado: 66 inicios falsos y 1 363 vueltas
  previas ficticias, todas sin rastrear. Eso daba el 58 % rastreado que se informó antes. Se corrigió en
  `ages_20261028.py` y al mismo tiempo se agregaron las sesiones sprint. Efecto [V]:
  - la proporción usada para completar (0,267 → 0,268) y la grilla θ–φ no cambian;
  - el cambio de SSE de la Ec. 1 cambia ≤ 0,01 puntos porcentuales;
  - cambian las edades `TL_Q` y `TL_GQ` de 1 846 vueltas: las sin compuesto, que la Ec. 1 y LogitStop descartan (la
    fórmula usa `C(Compound_Detail)`), y las de juegos con vueltas en sprint;
  - la cadena v10e (sección siguiente) se corrió con las edades anteriores. [H] Para v10eG no importa (`TL_G` cambia en
    14 vueltas sin compuesto); para v10eGQ habría que re-correrla si se adopta.
- **Ajuste:** la variante `GQ` reduce la suma de cuadrados de residuos frente a FastF1:

  | Conjunto | GQ | G | Q |
  |---|---|---|---|
  | 2021–23, 25 GP | −1,2 % (16/25 GP mejoran) | −0,5 % | −0,75 % |
  | 2021–23, 8 GP del paper | −2,2 % (7/8 GP) | −1,0 % | −1,3 % |
  | 2024, 21 GP | −0,6 % (15/21) | −0,3 % | −0,1 % |
  | 2024, 8 GP del paper | −0,5 % (6/8) | −0,35 % | −0,1 % |

  Con el mismo número de parámetros, −1,2 % en 54 472 vueltas equivale a 2·Δlog-verosimilitud ≈ 690.
- **Grilla:** el mínimo está en θ = 0–0,25 y φ = 0–0,25 en los cuatro conjuntos. Una vuelta bajo SC o VSC desgasta entre
  nada y un cuarto de una vuelta verde, lo que respalda el congelamiento actual del simulador. Las vueltas lentas previas
  casi no desgastan.
- **Pendientes:** la mediana 2021–23 sube de 0,048 a 0,052 s/vuelta; en 2024 casi no cambia.

## Qué faltaría para adoptarlo (no hecho)

- Re-estimar el factor 2021, que hoy se estima con la edad de FastF1.
- Re-estimar la cadena completa con la edad efectiva: neumáticos conjuntos, Ec. 1, LogitStop, K1 y adelantamiento.
- En el simulador: el congelamiento bajo SC/VSC ya es consistente. Habría que llevar a vueltas de empuje la edad de los
  juegos usados, tanto en el inventario 2024 como en las edades de tandas reales de los rivales.

## Cadena completa en las dos variantes (28-oct, prueba; nada adoptado)

- Cadenas: `results/sim_v10eG_20261028/` y `results/sim_v10eGQ_20261028/`, copias de la de v10d.
  - `age_fix_v10e_20261028.py` reemplaza `TyreLife` al leer los CSV.
  - Los pasos por el pit lane se detectan con la edad de FastF1.
- Factor 2021: `data/tyrelife_2021_factors_v10e{G,GQ}.json`.
- Pilotos: `results/driver_coef_v10_20261028/v10e{G,GQ}/`.
- Árboles: `trained_models/simulation{,_onedriver}_v10e{G,GQ}_20261028`, construidos con
  `scripts/build_simulation_v10e_20261028.py`.
- QMIP: `scripts/one_driver_v10e_20261028.py` con `F1_V10E=G|GQ`.
- Comparación: `chain_compare/` (`compare_chain_20261028.py`, `qmip_*.txt`).

Resultados frente a v10d [V]:

| | v10eG (sin SC/VSC) | v10eGQ (+ vueltas lentas previas) |
|---|---|---|
| Factor 2021 | 0.768 → 0.77 (física 0.680, conductual 0.860) | 0.774 → 0.77 (0.692, 0.872) |
| Pendientes de neumáticos, simulador, 8 GP | +3.7 % en promedio; US +13 %, Belgian +5 % | +11 % en promedio; Saudi +60 % (desde 0.010) |
| Desplazamientos entre compuestos | ≤ 0.05 s | ≤ 0.05 s |
| Ec. 1, otros coeficientes | intercepto y dummies de año ≤ 0.37 s (compensan el cambio de edad); combustible y pit ≤ 0.07 | ídem |
| LogitStop | pendientes de edad +3 a +7 %; intercepto −5.41 → −5.57 | pendientes +4 a +12 %; −5.60 |
| K1, SC/VSC | sin cambio | sin cambio |
| Adelantamiento | ≤ 0.017 | ≤ 0.017 |
| Pilotos (Ec. 2, 2024) | media 0.010, máximo 0.12 s/vuelta | media 0.017, máximo 0.07 s/vuelta |
| Fig. 4 (media con / sin corrección) | 0.193 → 0.192 / 0.636 → 0.632 | 0.190 / 0.625 |

Estrategias óptimas de la QMIP mono-piloto que cambian:

| GP | v10d | v10eG | v10eGQ |
|---|---|---|---|
| Bahrain | C3–C3–C2 | igual | C2–C2–C1 (dos tandas en C2 y una en C1) |
| Belgian | C4–C4–C3 | igual | C4–C4–C2 |
| Dutch | C3–C3–C2, paradas 25 y 49 | paradas 24 y 48 | ídem v10eG |
| Saudi | C3 → C4, parada 21 | C3 → C4, parada 26 (DP: 22) | C4 → C3, parada 26 |
| Otros 4 GP | — | igual | igual (US: segunda parada 37) |

- Saudi v10eG: la QMIP queda 0.21 s sobre el DP. [H] Está dentro de la tolerancia de Gurobi (MIPGap 1e-4 ≈ 0.47 s), no
  es un error de formulación.
- En v10eGQ falta, en el simulador, llevar a vueltas de empuje las edades de los juegos usados (inventario 2024 y tandas
  reales de los rivales).

## Peso de las vueltas de empuje de clasificación (28-oct)

Pedido del autor: probar un desgaste extra para las vueltas rápidas de clasificación, por ejemplo ×2. Grilla ψ ∈
{0, 0,5, 1, 1,5, 2, 3, 4} × θ ∈ {0, 0,25}, con el mismo procedimiento (`grid_quali_summary.txt`). Cambio de SSE frente a
FastF1 con θ = 0 [V]:

| Conjunto | ψ = 0 | ψ = 1 (GQ) | ψ = 2 | ψ = 4 | G | Mejor ψ |
|---|---|---|---|---|---|---|
| 2021–23, 25 GP | −1,28 % | −1,19 % | −0,91 % | +0,08 % | −0,50 % | 0 |
| 2021–23, 8 GP | −2,35 % | −2,17 % | −1,78 % | −0,51 % | −1,03 % | 0 |
| 2024, 21 GP | −0,42 % | −0,56 % | −0,57 % | −0,32 % | −0,31 % | 1,5 |
| 2024, 8 GP | −0,35 % | −0,52 % | −0,55 % | −0,31 % | −0,35 % | 1,5–2 |

- En 2021–23 el ajuste empeora a medida que ψ sube; en 2024 el mejor ψ está entre 1,5 y 2, pero la ventaja sobre ψ = 1
  es de 0,03 puntos. Los datos no respaldan un factor ×2 de forma consistente; ψ = 1 queda cerca del mejor en los cuatro
  conjuntos.
- [H] La diferencia entre períodos podría venir de la regla 2021 de largar con los neumáticos de Q2, de la selección (los
  equipos que llegan con juegos usados de clasificación) o de que la edad solo cambia el nivel de cada tanda, por lo que el
  peso se identifica débilmente.

## Adopción de v10eGQ (28-oct, decisión del autor)

1. **Cadena re-estimada** con las edades corregidas (sprint y compuestos NaN): factor 2021 0,774 (0,77, sin cambio;
   `data/tyrelife_2021_factors_v10eGQ.json`); neumáticos y adelantamiento cambian ≤ 0,002 frente a la corrida de la
   mañana; modelo SC/VSC sin cambio. Pilotos (Ec. 2): ≤ 0,017 s; Fig. 4: 0,190 / 0,625 (igual). La corrida anterior de
   la cadena y los árboles de prueba quedaron fuera del repo (carpeta temporal de la sesión).
2. **Edades del simulador en vueltas de empuje** (`sim_ages_gq_20261028.py` → `sim_ages_gq_summary.txt`):
   - `data/tyre_sets_2024_reg_v10eGQ_20261028.csv`: mismo inventario por reglamento que
     `data/tyre_sets_2024_reg_20261021.csv` (misma regla de devoluciones, que decide con las vueltas reales; se verifica
     que se reproduce exactamente), con la edad de cada juego usado = vueltas de empuje en las sesiones donde se usó
     (+ 0,268 por vuelta no encontrada; se encontraron todas). Edad media de los juegos usados 3,7 → 1,5.
   - `data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv`: edad de cada tanda real de 2024 = la misma edad de llegada que
     usa la estimación; 48 filas que son continuaciones de una tanda (vueltas faltantes) toman `TL_GQ` de la vuelta
     anterior. Edad media de las tandas con juego usado 2,1 → 0,55.
   - Las edades se redondean al entero (el simulador las guarda como enteros).
3. **Simulador:** `src/tyre_sets.py` lee la tabla de tandas reales de `F1_TYRE_STINT_AGES` (opcional; sin la variable,
   igual que antes). `scripts/tyre_ages_v10eGQ_20261028.py` fija `F1_TYRE_SETS` y `F1_TYRE_STINT_AGES` de v10eGQ y va
   primero en cada tarea. El congelamiento bajo SC/VSC ya era consistente con la edad G.
4. **Árboles:** `trained_models/simulation{,_onedriver}_v10eGQ_20261028` (reconstruidos). QMIP mono-piloto = DP en los 8
   GP (`chain_compare/qmip_v10eGQ.txt`).
5. **Tareas:** `reproduce/cluster/tasks_v10eGQ_20261028.tsv` (400) y `tasks_onedriver_v10eGQ_20261028.tsv` (200),
   `scripts/make_tasks_v10eGQ_20261028.py`; mono-piloto con `scripts/one_driver_v10eGQ_20261028.py`.
6. **Evaluación apuntando a v10eGQ:** `select_seeds_v5_20261010.py` (calib v10), `run_ds_v6_sweep_20261015.py` (v10,
   v10smhist), `make_tasks_v10sm_20261027.py` (→ `tasks_v10eGQsm_20261028.tsv`), `evaluate_world2024_20261010.py`
   (`F1_WORLD2024_TYRES=v10eGQ`), `reproduce/pipeline_v10_20261027.sh` (salida `results/world2024_v10eGQ_20261028`),
   `results/decision_support_v10eGQw24belief_20261028/run_20261028.sh`.
