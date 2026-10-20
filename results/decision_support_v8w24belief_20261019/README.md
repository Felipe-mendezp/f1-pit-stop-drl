# Apoyo a la decision v8, 8 GP (2026-10-19)

Corrida (19-oct, desde 22:28, con caffeinate): `zsh results/decision_support_v8w24belief_20261019/run_20261019.sh`
= `source reproduce/env_paper_v3_20261007.sh; F1_WORLD2024_TYRES=v7` + variables v8 (`F1_RIVAL_START_ALPHA`,
`F1_LAP0_ACTION0_MEDIUM=1`, `F1_LAP0_PENALTY=-1000`), sin F1_RIVAL_DNF, `python scripts/run_ds_v6_sweep_20261015.py
--version v8 --M 500 --workers 8 --figures` (opcion `--version v8` nueva). Agentes
`trained_models/rl_agents_v8_20261018/selected` (reglas 5/7/8 + `--max-dsq 0.01`), simulador `simulation_v7_20261015`,
mismos pilotos y protocolo que v7 (`../decision_support_v7w24belief_20261015/README.md`). Log `sweep_20261019.log`.
En v7 tomo ~4.5 h.

Despues: `summary_20261019.py` y `web_data_20261019.py` (copias de los de v7; solo cambian nombres de archivo) ->
`summary_20261019.csv`, `web/decisiones_v8_20261019.html` (plantilla `web/template_20261019.html`, textos v8).

## Resultados (20-oct 03:05, verificados en summary_20261019.csv)

- Barrido: 32/32 combinaciones rc=0 (terminado 20-oct 03:03, ~4.6 h); figuras y picks OK.
- Ninguna estrategia termina con un solo compuesto (32/32 sin DSQ).
- rec = mejor (v7 -> v8): media 94.2 -> 91.9 %, minimo 81 -> 62 % (PPO (ii) Hungarian; luego RPPO (ii) US 75 %).
  Regret medio 0.009 -> 0.01 posiciones, maximo 0.03 -> 0.05. "rec = no parar, mejor = parar": media 2.2 -> 3.8
  vueltas por combinacion, maximo 9 -> 25.
- Estrategias: 31/32 cambian respecto de v7 (otros agentes). Cambios visibles: Bahrain PPO parte con C2 y RPPO con C1
  (v7: C3); Saudi PPO y RPPO invierten el orden C3/C4; Singapore parte con C4 y para a C5 (v7: C5 -> C4); Miami PPO
  parte con C3 y para a C4; Hungarian PPO (ii) una parada (v7: dos).
- Singapore vuelta 18 (`singapore_lap18_20261019.csv`): no parar sigue siendo lo recomendado y lo mejor (PPO y RPPO, (i)
  y (ii)).
- Pagina: `web/decisiones_v8_20261019.html` (0.97 MB de datos).
- Publicada como artifact privado: https://claude.ai/artifact/VMXZv8PLCWAKhLchP7xAvC (20-oct).
