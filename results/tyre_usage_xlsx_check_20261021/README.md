# Revisión de `data/f1_2024_tire_usage.xlsx` contra FastF1 — 2026-10-21

Exploratorio (no toca el paper ni el simulador). El archivo tiene dos hojas de datos:
`Lapchive Sets` (6.490 juegos reconstruidos por Lapchive, 23 GP, sin Las Vegas) y `OpenF1 Stints`
(8.923 tandas, 24 GP, todas las sesiones, incluida la Sprint Qualifying).

- `check_tyre_usage_20261021.py` → `summary.txt`, `A_…csv` a `D_…csv`.
- `inventory_openf1_20261021.py` → `inventory_openf1_2024.csv`, `inventory_openf1_2024_validation.csv`.

## Resultados [verificado]

**OpenF1 vs FastF1** (8 GP, 2.739 tandas emparejadas; misma fuente, el live timing de F1):
- compuesto igual en el 99,7 %, vuelta inicial igual en el 99,6 %;
- edad inicial (`Edad al inicio` = TyreLife − 1) igual en el 92,7 %; de las 200 diferencias, 191 son tandas
  que OpenF1 marca como nuevas (edad 0) y FastF1 como usadas;
- la `Vuelta final` de OpenF1 repite la vuelta inicial de la tanda siguiente: en las tandas que no son la última de
  la sesión cuenta 1 vuelta de más (1.703 de 1.979).
- Carrera, 24 GP, contra `data/session_2024_V2.csv`: compuesto 99,5 %, edad inicial 95,1 %.

**Lapchive** (reconstrucción de terceros, sin ID físico; devoluciones "estimadas" a partir del reglamento):
- asignación (H, M, S) = (2, 3, 8) o (2, 4, 6) por piloto y GP: coherente con el reglamento;
- contra nuestro inventario FastF1 (8 GP, 418 celdas piloto × GP × compuesto): mismo número de juegos usados antes
  de la carrera en el 76 %, mismas vueltas juego a juego solo en el 3 %. Lapchive cuenta más: +1 vuelta por juego en
  la mayoría, +2 o +3 en juegos con varias tandas (hereda la vuelta duplicada de OpenF1), y hereda también los
  juegos "nuevos" espurios (el 5 % de los juegos fueron fusionados por Lapchive para cuadrar la asignación);
- tandas de carrera que parten con juego usado (edad ≥ 3): Lapchive tiene un juego con las vueltas exactas en el
  45 % (23 GP); en los 8 GP, 67 % Lapchive vs 81 % nuestro inventario FastF1.

**Inventario propio desde OpenF1** (misma regla que el inventario FastF1, corrigiendo la vuelta duplicada):
- partida de carrera con juego usado (edad ≥ 3): 21/21 exactas; tras pits 23/38, y las 15 que fallan son
  reutilizaciones dentro de la carrera (Imola ALB, Qatar vuelta 37 con tandas partidas), no del inventario previo;
- pero crea juegos de más (hasta 13 blandos, por encima de los 8 asignados) por los falsos "edad 0" de OpenF1, y
  solo coincide con el inventario FastF1 en el 50 % de las celdas.

## Conclusión

- La hoja OpenF1 concuerda con FastF1 (es la misma fuente) salvo los ~7 % de tandas con edad 0 errónea. Su
  aporte nuevo: Qualifying y Sprint Qualifying de los 24 GP (nuestra caché FastF1 solo tiene la Qualifying de 8 GP
  y ninguna Sprint Qualifying; las FP, Sprint y carrera de los 24 GP ya están en `data/`).
- La hoja Lapchive no sirve como fuente de vueltas por juego (sesgo de +1 vuelta por tanda y juegos fusionados);
  sí como control del número de juegos asignados y de los "nunca montados".
