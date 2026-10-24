# Variante v9r2 con --max-dsq 0 (2026-10-23, pedida por el autor; no reemplaza la principal)

Motivo: en la Fig. 7 principal, Saudi A2C (semilla 7) queda descalificado en el 60.8 % de las carreras: para en el SC
real de la vuelta 7 o no para nunca. Diagnostico por semilla en `../saudi_a2c_seeds_fig7/` (Fig. 7 Saudi A2C con las 10
semillas, `run_20261023.sh`): fallan las semillas 4, 7 y 8 (91, 61 y 69 % DSQ), que tenian 0.5-2 % DSQ en validacion; las
4 con 0 % en validacion funcionan (0-1.2 %). La regla 0 % usa solo la validacion, pero se propuso despues de ver la Fig. 7.

Hecho: `selection/` = copia de los val_*.csv de la seleccion principal + `seed_summary_..._r578_dsq0.csv`;
`agents/seedXX` = enlaces a trained_models/rl_agents_v9r2_20261022/seedXX, `agents/selected/` = copias de las elegidas.
    OUT=$A AGENTS=$A/agents SELOUT=$A/selection MAXDSQ=0 zsh reproduce/pipeline_v9r2_20261023.sh select   (A = este dir)
    OUT=$A AGENTS=$A/agents zsh reproduce/pipeline_v9r2_20261023.sh fig6|fig7 Saudi_Arabian_NOR Dutch_STR
Cambian 3 de 40: Saudi DQN 9->2, Saudi A2C 7->1, Dutch TRPO 4->3 (los 4 TRPO sin semilla con 0 %, Saudi/Miami/Singapore/US,
mantienen la eleccion). Los otros 6 GP se copiaron de la corrida principal (mismos agentes). Control: QMIP, PPO y RPPO de
Saudi y Dutch identicos a la corrida principal. Figuras `fig*/figure_*_2x4.pdf`, tabla `ci_table_v9r2_20261023.tex`.

Promedio 8 GP (diferencia con QMIP): Fig. 6 DQN 8.30 (+0.17), A2C 8.32 (+0.19), TRPO 8.58 (+0.44), PPO/RPPO sin cambio.
Fig. 7: A2C 9.62 -> 8.58 (+0.52 -> -0.52), DQN 9.46, TRPO 9.23, PPO 8.62, RPPO 8.78, QMIP 9.10. DSQ maximo por par en Fig. 7:
1.0 % (Bahrain DQN).
