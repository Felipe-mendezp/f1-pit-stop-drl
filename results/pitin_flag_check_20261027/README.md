# La columna PitIn/PitOut de session_*_V2.csv bajo SC (27-oct-2026)

> **Nota del 3-nov-2026:** este README describe el simulador anterior a v10. v10 elimina las vueltas de paso sin cambio de neumático (`05_yf_v4_drop_pass_20261027.py`; hoy k_in 0,366 y k_out 0,702 en `results/sim_v10eGQ_20261028/yf_model_v10_20261027.csv`), así que el "≈ 3 s" de abajo ya no aplica al simulador vigente.

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

## LogitStop a nivel de evento (diseño recordado por el autor, 27-oct) [V]

El autor recuerda haber calibrado LogitStop asignando todas las paradas de un evento SC/VSC a su primera vuelta. **El código
vigente no lo hace:** `src/models/rival_models.py`, que usa `results/sim_v7_20261015/rivals_v7_20261015.py` para el
simulador v7, estima un logit por vuelta con PitIn de FastF1. Tampoco aparece en los respaldos `Codigos/respaldo/
{11-12-2025,14-12-2025,30-09-2026 MAC Charles}/rivals_logit.py`.

`04_logitstop_event_level_20261027.py` lo implementa así: las vueltas verdes quedan igual; cada evento SC/VSC aporta una
fila por piloto, en su primera vuelta, con y = 1 si hubo cambio de juego en cualquier vuelta del evento.

| Modelo | Coef. SC | Prob. predicha, 1.ª v. SC | Coef. VSC | Prob. predicha, 1.ª v. VSC |
|---|---|---|---|---|
| Actual (por vuelta, PitIn) | 2,75 | 0,20 | 1,74 | 0,10 |
| Por evento (cambio de juego) | 3,44 | 0,30 | 2,35 | 0,16 |
| Real: proporción que para en el evento (2021–23) | | 0,35 | | 0,18 |

El simulador actual subestima las paradas de los rivales bajo SC/VSC. La versión por evento se acerca bastante a los datos
y además resuelve el problema del pit lane.
