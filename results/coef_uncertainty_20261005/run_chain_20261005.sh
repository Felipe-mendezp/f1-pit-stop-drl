#!/usr/bin/env bash
# 2026-10-05 — Cadena nocturna: Fig. 7 (publicados, f3 y f1) y Figs. 6-7 de los agentes Control (sin filtro de
# semilla, f3), con coeficientes de neumatico sorteados por carrera. Dos carriles en paralelo.
cd "$(dirname "$0")/../.."
source reproduce/env_paper.sh >/dev/null
export PYTHON=$HOME/.venvs/f1-drl-v2/bin/python MPLBACKEND=Agg
S=scripts/evaluate_coef_uncertainty_20261005.py; L=results/coef_uncertainty_20261005
CTRL=$PWD/trained_models/rl_agents_tyre_control_20261004
( $PYTHON $S --fig 7 --tag publicado --factor 3 > $L/fig7_publicado_f3.log 2>&1
  F1_RL_AGENTS_DIR=$CTRL $PYTHON $S --fig 6 --tag control --factor 3 > $L/fig6_control_f3.log 2>&1 ) &
( $PYTHON $S --fig 7 --tag publicado --factor 1 > $L/fig7_publicado_f1.log 2>&1
  F1_RL_AGENTS_DIR=$CTRL $PYTHON $S --fig 7 --tag control --factor 3 > $L/fig7_control_f3.log 2>&1 ) &
wait
echo "$(date '+%F %T') CHAIN DONE"
