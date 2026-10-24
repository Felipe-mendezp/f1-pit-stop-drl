# Apoyo a la decision v9r2, 8 GP (2026-10-23/24)

Corrida (23-oct 20:50 -> 24-oct 00:09, ~3.3 h, con caffeinate): `zsh results/decision_support_v9r2w24belief_20261023/run_20261023.sh`
= protocolo v8 (`../decision_support_v8w24belief_20261019/README.md`) + `F1_TYRE_SETS=data/tyre_sets_2024_reg_20261021.csv`,
`F1_TYRE_REUSE=1`, `F1_NO_SET_PENALTY=-1000`, `scripts/run_ds_v6_sweep_20261015.py --version v9r2` (opcion nueva). Agentes
`trained_models/rl_agents_v9r2_20261022/selected` (seleccion principal, --max-dsq 0.01; PPO y RPPO no cambian con la
variante --max-dsq 0). Prueba previa (Saudi PPO, vueltas 6-8, M = 10) en el tmp del job: rc = 0. Log `sweep_20261023.log`.
Despues: `summary_20261023.py`, `web_data_20261023.py` (copias de v8; solo nombres) -> `summary_20261023.csv`,
`web/decisiones_v9r2_20261023.html` (plantilla `web/template_20261023.html`, textos v9r2). Figuras por GP en `figs/<algo>/pdf/<GP>/`.

## Resultados (24-oct, de summary_20261023.csv) [V]
- 32/32 combinaciones rc = 0; figuras 4/4 rc = 0; ninguna estrategia termina con un solo compuesto.
- rec = mejor: media 94.6 % (v8 91.9), minimo 85 % (RPPO (ii) US; v8 62). Regret medio 0.009 posiciones (v8 0.014),
  maximo 0.03 (v8 0.05). "rec = no parar, mejor = parar": media 2.2 vueltas por combinacion (v8 3.8), maximo 8 (v8 25).
- Saudi NOR (caso de las Figs. 8-10 en v8): PPO y RPPO parten con C4 y paran a C3 en la vuelta 7 (SC), rec = mejor 100 %.
