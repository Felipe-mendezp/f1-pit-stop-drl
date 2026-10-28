# Compuestos 2021–2022 contra el live timing (28-oct-2026)

OpenF1 empieza en 2023, así que 2021–2022 se cotejan con el estado final de `TimingAppData` del live timing oficial
(espejo `livetiming-mirror.fastf1.dev`). Es el mismo origen que FastF1, pero no pasa por su asignación de compuesto por
vuelta, que es donde estaba el error de Bélgica 2024. Los casos con diferencias se confirmaron con una tercera fuente
(Pirelli vía Pitpass, FIA/F1 pit stop summary, RaceFans). El espejo no tiene 2023 ni Bélgica 2024.

- `fetch_livetiming_20261028.py 2021 2022` → `raw/<año>/<GP>/{TimingAppData,DriverList}.json` (44 carreras).
- `compare_20261028.py 2021 2022` → `mismatch_<año>.csv`, `mismatch_stints_<año>.csv` (por vuelta, sin vueltas de
  PitIn/PitOut) y `sequences_<año>.csv` (secuencia de compuestos por piloto; sirve cuando el feed registra mal los largos).

## Resultados [V]

| Año | Vueltas secas comparadas | Diferencias |
|---|---|---|
| 2021 | 20 497 | 0. Las 60 vueltas `UNKNOWN` (Bélgica, detrás del SC) son WET según el live timing |
| 2022 | 19 736 | Bélgica GAS y TSU, Mónaco LAT (abajo). Austria: artefacto del feed, no error de FastF1 |

Errores de FastF1 confirmados:

| Carrera | Piloto | FastF1 | Correcto | Confirmación |
|---|---|---|---|---|
| Bélgica 2022 | GAS | vueltas 23–44 SOFT (C4) | MEDIUM (C3) | live timing; Pirelli/Pitpass "NM, NH, NM" |
| Bélgica 2022 | TSU | vueltas 31–44 SOFT (C4) | MEDIUM (C3) | live timing; Pirelli/Pitpass "NH, NM, NM" |
| Mónaco 2022 | LAT | vueltas 3–19 HARD, edad 1–17; 20–22 HARD, edad 18–20 | 3–19 INTERMEDIATE; 20–22 HARD, edad 1–3 | live timing; FIA (parada en la 19); RaceFans ("pitted for slicks on lap 19") |

- **Mecanismo** (Bélgica 2022): el live timing tiene al final una tanda fantasma SOFT de 0 vueltas, y FastF1 la asigna
  a la última tanda real. Hay otros dos casos de tanda fantasma (Italia 2021 HAM, Holanda 2022 TSU) en los que FastF1
  acertó. [H] Bélgica 2024 sería lo mismo; no se puede comprobar porque el espejo no tiene esa carrera.
- **Sin compuesto en FastF1, se descartan con `dropna` (no sesgan):**
  - Austria 2022 desde ~la vuelta 40: el feed deja de actualizar las tandas; según el live timing es la misma tanda;
  - Mónaco 2022: STR 30–64 (HARD) y LAT 23–63.
- **Lluvia:** se filtra en la estimación, así que no importa.

Estado: **aplicado en v10d** (decisión del autor, 28-oct): `results/sim_v10d_20261028/data_fix_v10d_20261028.py`. Las
vueltas sin compuesto (Austria 2022, Mónaco 2022 STR y LAT 23–63) se dejan como están, descartadas.
