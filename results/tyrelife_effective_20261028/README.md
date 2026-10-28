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
