# Diagnóstico del exceso de adelantamientos (30-oct)

Pregunta del autor: ¿el simulador adelanta demasiado (Hungría 83 por carrera vs 69 real; total 573 vs 470)?

## Método

- `sim_pairs_20261030.py` + `run_sim_20261030.sh r28_k 100`: pares adyacentes en el simulador (mundo con ruido de la
  Ec. 2, σ 0,28), 100 carreras por GP, agente con una parada fija, sus pares excluidos. Guarda gap, diferencia de tiempos
  medios y parada de cada auto (vuelta de la decisión y vuelta de salida, por `stint_laps`; la edad del neumático no sirve
  con juegos reusados).
- Real: `real_pairs_ot_gap_20261029.pkl` = muestra de estimación del logit v10f (2021–23, L+1 sin SC/VSC,
  delta_total en [-30, 5]). Es la base de la cifra "470".
- `compare_20261030.py`, `compare_nopit_20261030.py` (solo pares sin parada), `pit_passes_20261030.txt`.

## Resultados [V]

1. **El exceso viene de los cambios de posición por paradas, no de los adelantamientos en pista.**
   Por carrera, suma 8 GP: en pista sim 301 vs real 326; asociados a paradas sim 272 vs real 145.
2. **El 145 real está subestimado por la muestra:** la muestra del logit excluye las vueltas con SC/VSC en L+1 y filas
   con datos faltantes, y con eso pierde la mayoría de las paradas. Paradas de rival en la muestra vs paradas por carrera
   en los datos: Dutch 10,7 de 41, Singapore 2 de 14,5, Saudi 4,3 de 30,7, Hungría 28,7 de 32,3. El simulador mide
   todas las paradas en verde. La comparación de los pasados por parada no es válida con esta muestra.
3. **En pista, el modelo está dentro de la variación real** (`pit_passes_20261030.txt`):

   | GP | Real 2021 / 22 / 23 | Sim media [p10, p90] |
   |---|---|---|
   | Bahrain | 58 / 48 / 35 | 51 [41, 62] |
   | Belgian | – / 67 / 60 | 44 [36, 55] |
   | Dutch | 16 / 24 / 61 | 34 [26, 43] |
   | Hungría | 21 / 72 / 33 | 33 [23, 41] |
   | Miami | – / 44 / 58 | 34 [26, 42] |
   | Saudi | 22 / 33 / 38 | 27 [20, 34] |
   | Singapore | – / 9 / 30 | 33 [23, 44] |
   | US | 22 / 53 / 40 | 45 [35, 55] |

4. Forma: el sim tiene más pares a ≤ 1 s (Hungría 39 % vs 22 %) pero menor tasa por par cercano (≤ 0,5 s: 0,15 vs 0,71);
   se compensan. Belgian y Miami quedan algo bajo lo real; Singapore algo sobre 2023.

La comparación del 29-oct (`sim_por_gp.txt`, "Singapore 2,6×, Dutch 1,7×") mezclaba ambas cosas y no debe usarse.
