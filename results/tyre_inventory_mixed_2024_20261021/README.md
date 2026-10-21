# Inventario mixto de juegos antes de la carrera, 24 GP de 2024 — 2026-10-21

Exploratorio (no toca el paper ni el simulador). Extiende `../tyre_inventory_2024_20261021/` (8 GP) a los 24 GP,
tras la revisión de `data/f1_2024_tire_usage.xlsx` (`../tyre_usage_xlsx_check_20261021/`).

- Fuentes: FP1–FP3 y Sprint = `data/practice_laps_2024_20260930.csv` (FastF1); Qualifying = caché FastF1 de los 24 GP
  (`load_q_sq_20261021.py` → `q_sq_2024.csv`; antes solo se habían cargado 8); Sprint Qualifying = hoja OpenF1 del
  xlsx (la API de F1 sigue respondiendo 401 el 21-oct), corrigiendo la vuelta que OpenF1 cuenta dos veces; carrera
  (solo para validar) = `data/session_2024_V2.csv`; asignación por compuesto = hoja Lapchive (usados + nunca montados).
- Regla (`inventory_mixed_20261021.py`): tanda con TyreLife inicial ≤ 1 abre un juego (el indicador FreshTyre de
  FastF1 está mal en 90 tandas, 68 en la Q de México); una tanda usada va al juego del mismo compuesto con la mayor
  vida que no supere su edad inicial (con 1 vuelta de tolerancia).
- Salidas: `inventory_mixed_2024.csv` (un juego usado por fila, con sus usos por sesión),
  `inventory_mixed_2024_summary.csv` (piloto × GP × compuesto: asignación, juegos usados, vueltas, juegos nuevos =
  asignación − usados), `inventory_mixed_2024_validation.csv`, `summary.txt`.

## Resultados [verificado]

- 4.239 juegos usados antes de la carrera en 24 GP; 33 "unmatched" (0,8 %; 21 en la FP1 de Imola, donde FastF1 y
  OpenF1 tienen edades incoherentes).
- Usados > asignación en 19 de 1.419 celdas (1,3 %), casi siempre 1 blando de más: tandas que la fuente marca como
  nuevas siendo reutilizadas (p. ej. Q de México: FastF1 y OpenF1 reinician la edad en cada tanda). Quedan marcadas
  (`NewSets < 0`), sin corregir.
- Validación con la carrera: de 70 tandas que parten con juego usado (edad ≥ 3), 53 calzan exacto con un juego previo
  a la carrera (21/21 en la partida) y 16 son juegos reutilizados dentro de la misma carrera; 1 sin explicar.
  Las tandas con edad 1–2 (170) calzan con un juego previo de 1–2 vueltas en el 74 % (95 % en la partida).
- Tandas de carrera (24 GP, compuestos secos): 80 % con juego nuevo, 14 % con 1–2 vueltas, 6 % con ≥ 3 vueltas
  (por compuesto, edad ≥ 3: blando 39 de 154, medio 20 de 489, duro 11 de 544).
- Juegos nuevos al llegar a la carrera (mediana por piloto): duro 1, medio 1, blando 1.

## Limitaciones

- No se observan las devoluciones obligatorias tras cada sesión: el inventario es de juegos *usados* antes de la
  carrera, no de juegos *disponibles*. Lapchive estima las devoluciones con el reglamento (no es dato oficial).
- Sprint Qualifying depende de OpenF1 (≈7 % de edades iniciales erróneas en las sesiones donde se pudo comparar).

## Juegos disponibles al inicio de la carrera (`available_at_race_20261021.py`)

- Devoluciones: fines de semana normales = columnas "Detalle de sesión" ("↩ FPx return") y "Retorno estimado" de
  Lapchive (estimación de Lapchive a partir del reglamento, no observación). Por piloto × GP × compuesto × sesión, los
  juegos devueltos se emparejan con nuestros juegos cuyo último uso antes de la carrera fue esa sesión: 1.202 de 1.373
  grupos directos (mismo número), 106 ambiguos (se eligen por cercanía de vueltas, empate = el más gastado), 65 con
  menos candidatos (65 devoluciones quedan sin juego nuestro). Las vueltas que quedan son las nuestras.
  Fines de semana sprint (Lapchive no marca devoluciones): regla propia, se devuelven los juegos usados solo en FP1.
- Asignación: (H, M, S) = (2, 3, 8) en fin de semana normal y (2, 4, 6) en sprint [verificado: es lo que Lapchive
  lista en los 353 y 120 casos]; Las Vegas se asume normal. Juegos nuevos = asignación − usados.
- Salidas: `available_at_race_2024.csv` (un juego disponible por fila; nuevos con 0 vueltas),
  `available_at_race_2024_summary.csv` (piloto × GP × compuesto: New, UsedKept, UsedLaps, Returned, Flags),
  `returns_matching_2024.csv`, `available_summary.txt`.
- Validación con la carrera: las 51 tandas que parten con un juego de ≥ 3 vueltas usan un juego que sigue disponible
  tras las devoluciones (100 %); con edad 1–2, 106 de 125 (85 %; las 19 que fallan son juegos de 1 vuelta, sobre todo
  Aston Martin, que [hipótesis] serían juegos nuevos con la vuelta de reconocimiento, no los de FP). Juegos nuevos usados en carrera
  > disponibles en 16 de 796 celdas antes de sumar los sustitutos, 17 después (2 %; FP1 de Imola, medios de Qatar, ver `NewSets < 0` del inventario).
- Promedio por piloto y GP (normal / sprint): duro nuevos 1,3 / 1,6 y usados 0,3 / 0; medio nuevos 1,3 / 1,3 y
  usados 0,4 / 2,1; blando nuevos 1,9 / 1,4 y usados 4,0 / 3,9.

### Umbral de no devolución (`available_at_race_20261021.py K`, salidas `*_keepK`)

Juegos con K vueltas o menos antes de la carrera nunca se devuelven [verificado, 24 GP; cifras con los sustitutos ya
sumados al titular de su auto, ver abajo]:

| K | devueltos | tandas de carrera con juego previo disponible | compuesto usado en carrera sin juego disponible | nuevos usados > disponibles |
|---|---|---|---|---|
| 0 | 1.723 | 157/176 (89 %) | 6 de 928 | 17 de 796 |
| 1 | 1.694 | 176/176 | 0 | 17 |
| 2 | 1.688 | 176/176 | 0 | 17 |
| 3 | 1.670 | 176/176 | 0 | 17 |

- K = 1 basta: rescata 29 juegos de 1 vuelta (los 19 fallos, casi todos Aston Martin, y los 6 compuestos sin juego);
  K = 2 o 3 conserva 6 y 24 juegos más sin mejorar ninguna validación.
- Explicación alternativa descartada: si las tandas de carrera con 1–2 vueltas fueran juegos nuevos con la vuelta de
  reconocimiento (K = 0), los juegos nuevos usados superarían a los disponibles en 58 celdas en vez de 16.
- Hay 167 juegos de 1 vuelta antes de la carrera (129 usados por última vez en FP3; ALO y STR 36 cada uno): juegos
  "asentados" con una vuelta para la carrera.

### Sustitutos (21-oct)

Una asignación de neumáticos por auto: cada piloto fuera de los 20 del simulador (novatos de FP1, BEA por SAI en Saudi,
COL por SAR, LAW por RIC, BEA por MAG, DOO por OCO) se suma al titular del auto que manejó (mismo equipo y ninguna
sesión en común, o titular ausente todo el fin de semana), en nuestros juegos, los de Lapchive y las vueltas de
carrera. Resultado: los 20 pilotos del simulador en los 24 GP, con al menos 1 juego de cada compuesto. Entrada del
simulador: `data/tyre_sets_2024_20261021.csv` y `data/tyre_race_stint_ages_2024_20261021.csv`
(`scripts/build_tyre_sets_20261021.py`, desde las salidas `_keep1`).

## Devoluciones por reglamento (`available_regulation_20261021.py`, `load_q3_20261021.py` -> `q3_2024.csv`)

Reglamento deportivo 2024 (art. 30.2, Issue 6) [verificado en el texto]: fin de semana normal, 13 juegos secos, se
devuelven 2 tras FP1, 2 tras FP2 y 2 tras FP3 (usados o no) y los que pasan a la Q3 devuelven uno de blando tras la
Qualifying -> 7 juegos en carrera (6 en Q3); sprint, 12 juegos, 1 tras FP1, 1 tras la sprint (el de más vueltas en
ella) y 3 tras la Qualifying -> 7. No se pueden devolver un duro y un medio (especificaciones de carrera) y, en fin de
semana normal, un blando no se puede usar ni devolver antes de la Q3 (quien no pasa a la Q3 lo tiene nuevo). Lapchive
solo marca devoluciones de juegos usados, por eso el inventario con Lapchive tiene 9,3 juegos por piloto (hasta 14).

Qué juego se devuelve no se observa (hipótesis). En cada devolución, entre los juegos no usados en sesiones
posteriores y no protegidos, se devuelve el de mayor puntaje = vueltas + w[compuesto]. Validación con las carreras
(240 tandas con juego usado, 796 celdas piloto x GP x compuesto con juegos nuevos usados):

| inventario | juegos por piloto | tandas con juego usado sin explicar | faltas de juego nuevo |
|---|---|---|---|
| Lapchive, K = 1 (actual) | 9,27 | 3 | 17 |
| reglamento, más vueltas primero, nuevos del compuesto que sobra | 6,73 | 11 | 23 |
| reglamento, menos vueltas primero | 6,73 | 43 | 23 |
| **reglamento, puntaje w = (blando 5, medio 0)** | 6,73 | **4** | **23** |
| reglamento, puntaje w = (blando 20 o más) | 6,73 | 3-4 | 39-61 |

- Los equipos devuelven primero blandos usados y guardan medios y duros poco usados (en sprint, los medios de la
  Sprint Qualifying se usan en carrera) y los blandos nuevos (castigos grandes al blando devuelven blandos nuevos que
  luego faltan en carrera).
- De las 23 faltas, 17 ya estaban con el inventario de Lapchive (problemas de datos: Qatar 5, Abu Dhabi 4, FP1 de
  Imola 4); 6 son propias del reglamento (3 pilotos que usaron 2 duros o medios nuevos, 3 blandos nuevos).
- 28 devoluciones no encuentran candidato (Canadá 12, México 14: todos los juegos se usan después); esos pilotos
  quedan con más de 7 juegos.
- Ejemplos: GAS Bélgica H [0,0] M [0] S [0,0,0,0] (antes 11 juegos, 6 blandos nuevos); juegos nuevos por piloto
  H 1,32 M 1,13 S 1,00 (antes 1,35 / 1,22 / 1,49).
- Especificaciones obligatorias de carrera [verificado, 21-oct]: en los 8 GP del paper la FIA declaró como obligatorias
  de carrera las dos más duras (duro y medio) y como neumático de Q3 el blando (tabla "Compounds selection" de los
  documentos FIA "Event Notes - Pirelli Preview" de cada GP, p. ej.
  https://api.fia.com/sites/default/files/decision-document/2024%20Bahrain%20Grand%20Prix%20-%20Event%20Notes%20-%20Pirelli%20Preview.pdf).
  Con tres compuestos, la regla de dos compuestos del simulador y del QMIP equivale entonces al art. 30.5 m), y la
  protección de un duro y un medio del inventario corresponde al art. 30.2 h ii).
