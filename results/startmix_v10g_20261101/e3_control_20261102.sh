#!/bin/zsh
# 2026-11-02 — E3 (control de olvido) de las 8 replicas elegidas del post-entrenamiento v10gsm: Figs. 6 y 7 con
# 500 simulaciones por GP, mismo mundo v10g y la misma QMIP. Usa reproduce/pipeline_v10g_20261031.sh con AGENTS
# apuntando al arbol post-entrenado y OUT a esta carpeta (los resultados de v10g en
# results/world2024_v10g_20261031 no se tocan). Equivalente a lo que se hizo en v9r2
# (results/startmix_proposal_20261025/e3_world2024_20261026).
#   zsh results/startmix_v10g_20261101/e3_control_20261102.sh
cd "$(dirname $0)/../.."
export OUT=results/startmix_v10g_20261101/e3_world2024_20261102
export AGENTS=trained_models/retrained_v10g_startmix_20261101
export ALGOS="PPO"
zsh reproduce/pipeline_v10g_20261031.sh fig6
zsh reproduce/pipeline_v10g_20261031.sh fig7
