# Edad efectiva de los neumáticos: prueba preliminar de la Ec. 1 (28-oct-2026)

Pedido del autor: que no cuenten como vuelta de uso (i) las vueltas bajo SC o VSC y (ii) las vueltas lentas de prácticas
y clasificación. Hoy los datos (FastF1) cuentan todas las vueltas; el simulador multi-piloto congela la edad bajo SC/VSC.

1. `extract_fpq_20261028.py` (entorno `~/.venvs/f1-fastf1`, caché local offline) → `fpq_laps_2021_2024.csv.gz`:
   vueltas de prácticas, clasificación y sprint 2021–2024.
2. `ages_20261028.py` → `ages_2021_2024.csv.gz`, `stint_starts.csv`, `fpq_stints.csv`. Variantes de edad:
   - `G`: sin vueltas de carrera con SC, VSC o bandera roja;
   - `Q`: la edad con que llega un juego usado cuenta solo vueltas de empuje (≤ 1,10 × la mejor del piloto en la
     sesión, sin entrada, salida ni bandera), siguiendo la cadena de tandas del juego en prácticas y clasificación;
   - `GQ`: ambas.
3. `compare_20261028.py`: la Ec. 1 por GP, con la misma muestra y fórmula que hoy (datos v10d, filtro A, factor 2021
   de 0.74), solo cambiando `TyreLife`.
4. `grid_20261028.py`: grilla de pesos `TyreLife = GQ + θ·(vueltas SC/VSC) + φ·(vueltas lentas previas)`.

## Resultados [V]

- **Juegos usados:** 21 % de las tandas de carrera. En los juegos de la largada se encuentra el 93 % de las vueltas previas;
  en total, el 58 %, por vueltas sin compuesto en Austria y Mónaco 2022, que igual se descartan. El resto se completa con
  la proporción global de vueltas de empuje (0,27). La edad media con que llega un juego usado baja de 4,0 a 1,1 vueltas.
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
