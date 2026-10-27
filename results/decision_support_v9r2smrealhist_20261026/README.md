# E4: barrido B (historia real) con los PPO/RPPO post-entrenados con largada sorteada — 2026-10-26

Agentes `trained_models/retrained_v9r2_startmix_20261025/selected` (E2: regla 5/7/8 + `--max-dsq 0` entre 3 réplicas;
E1 en la largada real, ver `../startmix_proposal_20261025/README.md`). Mismo protocolo que
`../decision_support_v9r2realhist_20261024/` (M = 500, PPO y RPPO, (i) y (ii), 8 GP, vueltas 1..N-1), versión
`v9r2smhist` de `scripts/run_ds_v6_sweep_20261015.py`, `F1_START_MIX` apagado. Lanzador `run_20261026.sh` (E4 y luego
E3), log `sweep_20261026.log`: 08:40–11:54, rc = 0 en las 32 corridas y en las figuras.

## Control de máscara (`mask_check_20261026.py` -> `mask_check_20261026.{csv,log}`, `mask_check_laps_20261026.csv`) [V]

| Barrido | rec = mejor (puntos) | pérdida media | rollouts de la acción recomendada corregidos |
|---|---|---|---|
| v9r2 original (trayectoria propia) | 94,5 % | 0,013 pts | 0 % |
| B, agentes originales | 93,4 % | 0,059 pts | 9,9 % |
| **B, agentes post-entrenados (Bsm)** | **94,3 %** | **0,030 pts** | **2,2 %** |

- Saudi: máscara de la acción recomendada 75 % -> 0–0,3 % (PPO) y 0,7–4,8 % (RPPO). Pasa el criterio E4 (<= 5 %).
- US RPPO: 31 % -> 31 %, sin cambio. No pasa (E1 ya lo anticipaba: 3 % de DSQ con la largada real). US PPO: 0,4 %.
- Resto: <= 1,3 % (Miami RPPO (ii)); antes 0 %.
- rec = mejor >= 93 % en el promedio; por GP sigue bajo en Bahrain (75–80 %) y Belgian RPPO (ii) (77 %), errores de
  elección sin máscara, como con los agentes originales.

## Saudi vuelta 7 (Norris, medios, P6, no paró bajo el SC) [V]

Puntos medios por acción, PPO (i) / (ii): blandos 15,4 / 13,8; duros 13,9 / 12,4; no parar 12,4 / 11,1; medios 11,7 / 9,9.
RPPO (i) / (ii): blandos 15,3 / 13,8; duros 13,7 / 12,3; no parar 12,7 / 11,6; medios 12,9 / 10,9.
La política recomienda blandos (= mejor) y la máscara corrige 0 % (PPO) y 7–30 % (RPPO, "no parar" y medios) de los
rollouts, frente a 97–100 % con los agentes originales. Parar a blandos supera a quedarse por 3,0 / 2,7 pts (PPO) y
2,6 / 2,2 (RPPO); con los originales la diferencia era 3,2 a 3,6 pero no era limpia. Sin pruebas de significancia aún.
