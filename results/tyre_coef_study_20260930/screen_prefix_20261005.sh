#!/usr/bin/env bash
# 2026-10-05 — Filtro de paradas de la publicacion (1-4 paradas, primera despues de la vuelta 4;
# README "Provenance of the agents") sobre publicados, control, opcion2 y opcion3, PPO y RPPO, replay 2024.
# Salida: results/tyre_coef_study_20260930/screen_prefix_20261005.csv
cd "$(dirname "$0")/../.."
PY_ENV="${PYTHON:-$HOME/.venvs/f1-drl-v2/bin/python}"
source reproduce/env_paper.sh >/dev/null; export PYTHON="$PY_ENV" MPLBACKEND=Agg
OUT=results/tyre_coef_study_20260930/screen_prefix_20261005.csv
echo "set,algo,model_dir,pits,n_laps" > $OUT
GPS="Bahrain Grand Prix|Driver_ALO|Bahrain_ALO
Saudi Arabian Grand Prix|Driver_NOR|Saudi_Arabian_NOR
Miami Grand Prix|Driver_OCO|Miami_OCO
Hungarian Grand Prix|Driver_SAI|Hungarian_SAI
Belgian Grand Prix|Driver_NOR|Belgian_NOR
Dutch Grand Prix|Driver_RUS|Dutch_RUS
Singapore Grand Prix|Driver_HUL|Singapore_HUL
United States Grand Prix|Driver_PIA|United_States_PIA"
for set in actual control opcion2 opcion3; do
  case $set in
    actual)  sim=simulation; ag=rl_agents;;
    control) sim=simulation; ag=rl_agents_tyre_control_20261004;;
    *)       sim=simulation_tyre_${set}_20261004; ag=rl_agents_tyre_${set}_20261004;;
  esac
  export F1_SIMULATION_DIR=$PWD/trained_models/$sim F1_RL_AGENTS_DIR=$PWD/trained_models/$ag
  while IFS='|' read gp drv md; do for algo in PPO RecurrentPPO; do
    r=$($PYTHON scripts/diag_prefix_pits_algo_20261005.py --gp "$gp" --driver $drv --model-dir $md \
        --algo $algo --real-2024 --following-gap-scale 1.0 2>/dev/null | tail -1)
    pits=$(echo "$r" | sed -n 's/^pits=\[\(.*\)\] n_laps=\([0-9]*\)$/\1/p'); n=$(echo "$r" | sed -n 's/.*n_laps=//p')
    echo "$set,$algo,$md,\"$pits\",$n" | tee -a $OUT
  done; done <<< "$GPS"
done
