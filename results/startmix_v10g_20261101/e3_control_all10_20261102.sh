#!/bin/zsh
# 2026-11-02 (noche) — E3 (control de olvido) del conjunto FINAL de agentes v10gsm: las elegidas con 10 semillas por
# carrera (e2_all10_20261102) y con EE.UU. cambiado a la semilla 2 porque la regla ahora exige tambien el control E1.
# Salida aparte de e3_world2024_20261102, que midio las elegidas de 3 semillas.
#   zsh results/startmix_v10g_20261101/e3_control_all10_20261102.sh
cd "$(dirname $0)/../.."
export OUT=results/startmix_v10g_20261101/e3_world2024_all10_20261102
export AGENTS=trained_models/retrained_v10g_startmix_20261101
export ALGOS="PPO"
zsh reproduce/pipeline_v10g_20261031.sh fig6
zsh reproduce/pipeline_v10g_20261031.sh fig7
