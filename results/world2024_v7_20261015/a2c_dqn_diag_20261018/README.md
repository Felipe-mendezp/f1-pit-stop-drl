# Dutch A2C / US A2C (descalificaciones Fig. 7) y US DQN (paradas tardias Fig. 6), v7 reglas 5/7/8 — 2026-10-18

Archivos: `run_20261018.sh` (trazas con `../bahrain_trpo_dsq/diag_20261018.py`, 200 carreras; Fig. 7 de A2C con las 10
semillas en `seeds/`), `diag_dqn_20261018.py` (misma traza, intercepta tambien `DQN.predict`; trazas `trace/us_dqn_*b`),
`lap0_action_20261018.py` (accion en la v0 de los 400 modelos, 50 resets con rivales uniformes; `lap0/`).

## A2C Dutch (semilla 9) y US (semilla 7): mismo mecanismo que Bahrain TRPO [V]
- En la v0 eligen siempre la accion 0 (no valida): el entorno sortea el compuesto de partida (`_handle_lap_0`,
  penalizacion -600) en Fig. 6 y Fig. 7.
- Fig. 6 (rivales uniformes): la parada va al compuesto que corresponde al montado (Dutch: desde C1 a C3, si no a C1;
  US: a C2 o C4) -> 3.5 % / 2 % DSQ.
- Fig. 7 (rivales reales: Dutch 15 de 19 en C2, US 14 en C3): la parada va casi siempre a C1 (Dutch, 185/200) o C2
  (US, 192/200) sea cual sea el montado -> DSQ cuando el sorteo de partida coincide: Dutch 72.7 % de las que parten
  con C1 (20 % total), US 85.5 % de las que parten con C2 (23.5 %). Es decir, la politica elige el destino de la parada
  segun la composicion de compuestos de los rivales, que en la Fig. 7 queda fuera de lo visto.
- Fig. 7 por semilla (200 carreras, pos / % DSQ): Dutch 9.72/5.5, 11.46/22, 10.65/11, 9.17/1, 10.90/14.5, 10.46/8,
  12.38/31, 9.84/5.5, **11.36/20 (elegida)**, 12.11/27.5 (QMIP 9.56); US 9.88/3, 11.57/21, 9.39/2, 9.04/0, 9.70/6,
  10.05/3.5, **11.88/23.5 (elegida)**, 9.21/0, 11.84/25, 9.53/1.5 (QMIP 8.86). La DSQ de validacion no predice la de la
  Fig. 7 (US: todas 0-6.5 % en validacion; Dutch s9 4.5 % en validacion, 20 % en Fig. 7).
- Dutch A2C en general es malo: DSQ de validacion 4.5-51 % segun semilla, < 1 parada media.

## Accion 0 en la v0, 400 modelos [V]
Eligen la accion 0 (> 50 % de 50 resets con rivales uniformes): A2C 10/10 semillas en Bahrain, Dutch, Hungarian y US
(0/10 en los otros 4); PPO 6/10 en Bahrain y 1/10 en US; DQN, RPPO, TRPO 0. Total 47/400; 25 modelos mixtos.
Seleccionados: A2C Hungarian s6, Dutch s9, US s7, Bahrain s2 (4 de 40). Ademas TRPO Bahrain s4/7/9 eligen la accion 0
solo con los rivales reales de la Fig. 7 (0 % con rivales uniformes; ver `../bahrain_trpo_dsq/`).

## US DQN (semilla 7): paradas tardias en Fig. 6 [V]
- Parte siempre con C4 (accion 3), primera parada a C2 (vuelta 6-30, media ~24). En el 25.5 % de la Fig. 6 vuelve a parar
  en las vueltas 51-55 (34 de 51 en la 54 de 56), casi siempre a C3 (blando nuevo), con la regla ya cumplida; 22 % bajo
  SC/VSC.
- Disparador: TyreLife del C2 al final: P(parada tardia) 3 % con TyreLife <= 20 en la v53, 13 % en 21-25, 24 % en 26-28,
  83 % en 29-31, 100 % > 31. Es decir, depende de lo temprano de la primera parada.
- Cuesta posiciones: de la v53 (o la vuelta de la parada) al final pierde 1.8 posiciones sin SC (2.7 con SC) vs 0.85 en
  las carreras sin parada tardia; Fig. 6: 11.23 con parada tardia vs 8.07 sin ella.
- Validacion vs Fig. 6: primera parada media 26.9 vs 23.9; P(primera <= 24) 0.23 vs 0.39; parada tardia 7 % vs 25.8 %
  (21 % vs 42 % si la primera es <= 24). La semilla 7 pasa el filtro justo (share_ok 0.950).
- Fig. 7: 2 % (la primera parada es mas tardia, ~28).
- No tiene que ver con los compuestos de partida de los rivales: es una politica mala de DQN (parada para "neumatico
  fresco" al final que no compensa) que la validacion casi no ve. v8 no la corrige [H].
