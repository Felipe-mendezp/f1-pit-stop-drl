# La columna PitIn/PitOut de session_*_V2.csv bajo SC (27-oct-2026)

Diagnóstico, **sin cambios en el simulador ni en el paper**. Una parada "real" se define por el cambio de juego (baja
TyreLife o cambia Compound_Detail entre la vuelta L y L+1), como `real_plan`.

## El problema [V]
`PitIn = PitInTime no nulo` de FastF1 (`scripts/build_dataframes.py`). **No es un error del posprocesamiento ni de FastF1.**
En FastF1 3.8.3 crudo, Portugal 2021 (HAM, VER, NOR) trae `PitInTime` en las vueltas 2–3, `PitOutTime` en las 3–4 y
`Stint` 1 → 2 → 3, igual que el CSV. Los mensajes de dirección de carrera lo explican: "SAFETY CAR THROUGH THE PIT LANE"
(Portugal 2021, v. 2; São Paulo 2021, v. 7). El pelotón entero pasó por el pit lane detrás del SC, sin parar. Las banderas
registran pasos reales por el pit lane, pero **no son paradas**. Qatar 2024 no se pudo bajar de FastF1 (falla la carga de
la sesión), así que ese caso queda sin verificar [H].

`01_pitin_vs_stints_20261027.py`: entre las vueltas con PitIn = 1, la proporción con parada real es 95 % en verde, 97 %
bajo VSC y **63 % bajo SC**. En 2021–23, bajo SC, la tasa de PitIn es 11,7 % por vuelta y la de paradas reales 7,4 %.

## Dónde se usa y efecto [V]
1. **Modelo de tiempos YF v4** (`02_yf_v4_refit_20261027.py`): en su muestra, 64 de 98 vueltas con PitIn y 36 de 99 con
   PitOut no tienen parada.
   - Con las banderas corregidas, k_in casi no cambia (0,38 → 0,40) y **k_out sube de 0,62 a 0,78** (±0,09).
   - El ritmo bajo SC y bajo VSC no cambia.
   - En los 8 GP, la pérdida de una parada bajo SC/VSC sube 2,2–3,3 s (media 2,9 s), por ejemplo de 13,1 a 16,1 s en
     Saudi. Es decir, en el simulador parar bajo SC/VSC sale ≈ 3 s más barato de lo que indican los datos corregidos.
2. **LogitStop de los rivales** (`03_logitstop_refit_20261027.py`): PitIn es la variable dependiente.
   - Con las banderas corregidas, el coeficiente SC baja de 2,75 a 2,32 (el paper reporta 2,79); VSC casi igual.
   - Pero en la 1.ª vuelta de SC (la única en que el simulador deja parar a los rivales), la probabilidad media predicha es
     0,21 con el modelo actual y 0,15 corregido. En los datos 2021–23, la tasa real de paradas en esa vuelta es **0,29**.
   - El error de la columna **compensa en parte** la regla de "solo la 1.ª vuelta". Corregirlo alejaría al simulador de
     la realidad.
3. **Regresión de tiempos en verde** (Eq. 1): usa vueltas verdes, donde PitIn coincide con las paradas en el 95 %. Sin
   problema relevante [H, no reestimada].
