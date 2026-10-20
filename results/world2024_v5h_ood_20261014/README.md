# Paradas sin sentido de los DRL en la Fig. 7 v5h: abandonos de rivales (2026-10-14)

Hipotesis: un rival que abandona (estrategia real con -1) aparece en la observacion con `time_diff_to_agent = +500`
(cota del espacio). En el entrenamiento y en la Fig. 6 los rivales nunca abandonan. Tras VecNormalize, +500 queda a
z = 4-19 desviaciones (RPPO Saudi: 12-19, recortado a clip_obs = 10).

Intervenciones (`intervene_20261014.py`, Fig. 7, 200 carreras, mismas semillas que `../world2024_v5h_pits_20261014`):
- `nodnf`: el rival no abandona (sigue con su ultimo neumatico).
- `dnfobs`: el entorno no cambia; el agente ve el ultimo gap del rival antes de abandonar en vez de +500.

Resultado verificado (`summarize_20261014.py`):
- RPPO Saudi: % de carreras con parada en las ultimas 6 vueltas 56.5 -> 0.5 (dnfobs); posicion 7.22 -> 5.63.
- TRPO Singapore: 99 -> 3 (dnfobs); paradas 4.51 -> 1.22; posicion 14.14 -> 8.70.
- TRPO Saudi: 63 -> 11.5 con nodnf, pero 86.5 con dnfobs (sensible al abandono en ambas representaciones).
- No lo explica: DQN Saudi (3 paradas en ambas) ni A2C/TRPO Miami (~2.7); DQN Miami baja solo 4.5 -> 3.5.
- QMIP sin cambios en dnfobs (no usa la observacion).
Abandonos 2024 en la config: Saudi STR v7, GAS v2; Miami SAR v29; Singapore ALB v16, MAG v58; Bahrain ninguno.
