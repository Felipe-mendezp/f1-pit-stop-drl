# Bahrain TRPO, descalificaciones en la Fig. 7 v7 (reglas 5/7/8) — diagnóstico 2026-10-18

Agente: TRPO Bahrain_ALO, semilla 5 (elegida con las reglas 5/7/8; antes semilla 1). Fig. 7 v7: 96.8 % de carreras sin
cambio de compuesto (posición 20). Archivos: `diag_20261018.py` (envoltorio de `scripts/evaluate_world2024_20261010.py`
sin cambios en la simulación; traza cada paso del agente DRL desde la v0 y los compuestos de los rivales en v0-v1),
`run_20261018.sh` (500 carreras: trazas s05/s01 Fig. 7, s05 Fig. 6; Fig. 7 con cada semilla en `seeds/`),
`run2_20261018.sh` (100 carreras, `short/`: semillas 3, 4, 5, 7, 9, 10).

Resultados [V]:
- La semilla 5 para siempre a C2 (medio), sea cual sea el compuesto montado. Si parte con C3 cumple la regla; si
  parte con C2, no. Fig. 6: parte con C3 en 472/500 y con C2 en 28/500 (= sus 5.6 % de descalificaciones).
  Fig. 7: parte con C2 en 500/500.
- Lo que cambia en la v0: los compuestos de partida de los rivales. En entrenamiento, validación y Fig. 6 cada rival
  parte con un compuesto sorteado uniforme entre los 3 (`f1_env_all_drivers.py:697-700`; media 6.1/6.4/6.5 en C1/C2/C3);
  en la Fig. 7 con el real 2024 (los 19 rivales con C3). Es un estado de la v0 que el agente no vio al entrenar.
- Fig. 7 Bahrain TRPO por semilla (500 carreras; pos. media, % descalificado): 1: 6.68, 0 | 2: 7.02, 0 | 3: 20.0, 96 |
  4: 10.86, 33 | 5: 20.0, 97 | 6: 7.81, 0 | 7: 11.02, 34 | 8: 6.32, 0 | 9: 10.47, 33 | 10: 6.34, 0. QMIP 7.13.
  Las 5 semillas que pasan 5/7/8 (3, 4, 5, 7, 9) fallan; las 5 que no pasan (1, 2, 6, 8, 10) no.
- Semilla 3: igual que la 5 (parte con C2, para a C2). Semillas 4, 7, 9: en la v0 de la Fig. 7 eligen la acción 0
  (no válida); el entorno sortea el compuesto de partida (`_handle_lap_0`) y la parada va siempre a un compuesto fijo
  (C2, C3, C1), así que quedan descalificadas cuando el sorteo coincide con ese compuesto (~1/3).
- En la validación de la selección la semilla 5 tenía 4 % de descalificaciones (las otras 0-0.5 %, salvo 3: 8.5 % y
  9: 6 %); el filtro no las excluye (cuentan como posición 20 en la media). La semilla 4 tenía 0 % y en la Fig. 7 33 %:
  un filtro por descalificaciones en validación no lo habría evitado.
- Otros casos con descalificaciones en la Fig. 7 (iguales con ambas selecciones): Dutch A2C 24.8 %, US A2C 27.8 %
  (Fig. 6: 3.2 y 1.8 %). [H] probablemente el mismo mecanismo; no trazado.

Interpretación [H]: la falla no es de la regla 5/7/8 sino del sorteo uniforme de compuestos de partida de los rivales
en el simulador: algunas políticas aprenden una parada a compuesto fijo y un compuesto de partida que depende de los
rivales; con rivales reales (concentrados en un compuesto) caen fuera de lo visto. Que las 5 semillas que pasan fallen
y las 5 que no pasan no fallen puede ser casualidad (10 semillas).
