#!/bin/zsh
# 2026-11-02 — Medicion de las Figs. 6 y 7 en Miami con la semilla 7 del post-entrenamiento, la unica de las que
# pasan la regla de seleccion que cumple el criterio E1 amplio (0 % de DSQ con los cuatro compuestos de largada;
# la elegida por menor pos_media, la 8, descalifica en el 100 % de las carreras si se le impone C4, ver
# results/startmix_v10g_20261101/e1_20261102/e1_miami_seeds_20261102.csv). Salida aparte para no pisar lo ya
# medido con la semilla 8 en e3_world2024_20261102/.
#   zsh results/startmix_v10g_20261101/e3_miami_s7_20261102.sh
cd "$(dirname $0)/../.."
export OUT=results/startmix_v10g_20261101/e3_miami_s7_20261102
export AGENTS=results/startmix_v10g_20261101/alt_miami_s7_20261102
export ALGOS="PPO"
zsh reproduce/pipeline_v10g_20261031.sh fig6 Miami_OCO
zsh reproduce/pipeline_v10g_20261031.sh fig7 Miami_OCO
