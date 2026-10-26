#!/bin/zsh
# 2026-10-24: barridos de apoyo a la decision v9r2 con el prefijo forzado (pedido del autor):
#   A = realstart (largada real 2024 del piloto, despues la politica) -> results/decision_support_v9r2realstart_20261024/
#   B = realhist  (largada y paradas reales hasta la vuelta de decision) -> results/decision_support_v9r2realhist_20261024/
# Mismo protocolo que results/decision_support_v9r2w24belief_20261023/run_20261023.sh (M = 500, PPO y RPPO, (i) y (ii)).
# A solo corre los pares cuya largada difiere de la del agente (PPO: Saudi, Bahrain; RPPO: Saudi, Bahrain, US); los demas
# son identicos por construccion (misma accion en la vuelta 0, prefijo determinista) y se copian del barrido v9r2.
#   zsh results/decision_support_v9r2realhist_20261024/run_20261024.sh [A|B|AB]
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
SW=scripts/run_ds_v6_sweep_20261015.py
WHAT=${1:-AB}
if [[ $WHAT == *A* ]]; then
  export F1_DS_FORCE=start; A=results/decision_support_v9r2realstart_20261024; S=results/decision_support_v9r2w24belief_20261023
  echo "$(date +%T) A inicio"
  $PY $SW --version v9r2start --M 500 --workers 8 --algos PPO --gps Saudi_Arabian_NOR Bahrain_ALO
  $PY $SW --version v9r2start --M 500 --workers 8 --algos RecurrentPPO --gps Saudi_Arabian_NOR Bahrain_ALO United_States_MAG
  for algo k in PPO belgian_gas PPO dutch_str PPO hungarian_tsu PPO miami_oco PPO singapore_hul PPO united_states_mag \
                RecurrentPPO belgian_gas RecurrentPPO dutch_str RecurrentPPO hungarian_tsu RecurrentPPO miami_oco RecurrentPPO singapore_hul; do
    cp $S/$algo/${k}_real*_lap* $A/$algo/
  done
  echo "$(date +%T) A copiados los pares sin cambio"
  $PY $SW --version v9r2start --figures-only
  echo "$(date +%T) A fin"
fi
if [[ $WHAT == *B* ]]; then
  export F1_DS_FORCE=hist
  echo "$(date +%T) B inicio"
  $PY $SW --version v9r2hist --M 500 --workers 8 --figures
  echo "$(date +%T) B fin"
fi
