# Inventario de juegos de neumáticos antes de la carrera, 2024 (8 GP del paper) — 2026-10-21

Exploratorio (no está en el paper). Pregunta del autor: qué juegos tiene cada piloto justo antes de la carrera y con
cuántas vueltas.

- Fuentes: FP1–FP3 y Sprint de `data/practice_laps_2024_20260930.csv`; Qualifying desde la caché local de FastF1
  (`~/.cache/fastf1`, modo sin conexión, `load_quali_20261021.py` con `~/.venvs/f1-fastf1`); carrera de
  `data/session_2024_V2.csv` (tandas = vuelta 1 y salidas de pits). La API de tiempos de F1 responde 401 (21-oct),
  así que no se pudo descargar nada nuevo: **falta la Sprint Qualifying de Miami y US**.
- Orden: `load_quali_20261021.py` -> `build_laps_20261021.py` -> `inventory_20261021.py laps_2024_8gp.csv` (desde esta carpeta).
- Método (`inventory_20261021.py`): TyreLife de FastF1 arrastra las vueltas de sesiones previas; una tanda con
  FreshTyre abre un juego, una usada se asigna al juego del mismo compuesto con la mayor vida previa que no supere su
  vida inicial.
- Salidas: `inventory_2024_8gp.csv` (un juego por fila: compuesto, sesión en que se abrió, vueltas antes de la
  carrera, usos por sesión) y `inventory_validation_2024_8gp.csv` (cada tanda de carrera con juego usado y si calza).
- Validación: 43 de 50 tandas de carrera con juego usado calzan con un juego reconstruido, siempre con 1 vuelta de
  diferencia (la vuelta siguiente); 6 de los 7 que no calzan son de Miami (juegos medios usados en la Sprint
  Qualifying que falta). 37 de 1291 juegos quedan "unmatched" (usados sin registro previo): 34 en Miami y US.
- No incluye: juegos devueltos tras cada sesión (no están en los datos) ni sesiones con intermedios/lluvia.
