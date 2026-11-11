#!/bin/zsh
# 2026-11-11 — Verificacion independiente de la columna "mejor de las cuatro" (figura de Bahrein, pedido del autor).
# Repite, en las 28 vueltas en que la recomendacion PPO difiere de la decision real (escenario ii = --real-2024,
# compare_laps_20261103.csv), las 4 acciones con M = 500 y semillas de rollout DISJUNTAS de las del barrido
# (--base-seed $BASE en vez de 1000; mismo --prefix-seed 0, es decir el mismo estado en la vuelta de decision).
# Mismo entorno y mismos argumentos que run_20261102.sh / run_ds_v6_sweep_20261015.py --version v10gsmhist.
#   zsh results/decision_support_v10gsmrealhist_20261102/verif_best_20261111/run_verif_20261111.sh [PRUEBA]
cd "$(dirname $0)/../../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
export F1_DS_FORCE=hist OMP_NUM_THREADS=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_startmix_20261101/selected
unset F1_RIVAL_DNF F1_START_MIX
OUT=results/decision_support_v10gsmrealhist_20261102/verif_best_20261111
M=500
typeset -A LAPS GPN DRV
LAPS=(Bahrain_ALO "15 35 36 37 38 39 40" Belgian_GAS "9 25 26 27 28" Dutch_STR "30" Hungarian_TSU "29"
      Miami_OCO "22" Saudi_Arabian_NOR "7 29 31 32 33 34 35 36" Singapore_HUL "29" United_States_MAG "3 17 36 37")
GPN=(Bahrain_ALO "Bahrain Grand Prix" Belgian_GAS "Belgian Grand Prix" Dutch_STR "Dutch Grand Prix"
     Hungarian_TSU "Hungarian Grand Prix" Miami_OCO "Miami Grand Prix" Saudi_Arabian_NOR "Saudi Arabian Grand Prix"
     Singapore_HUL "Singapore Grand Prix" United_States_MAG "United States Grand Prix")
MDS=(${(k)LAPS})
BASE=100000
if [[ "$1" == PRUEBA ]]; then M=2; BASE=1000; MDS=(Bahrain_ALO); LAPS[Bahrain_ALO]="39"; OUT=$OUT/prueba; fi   # prueba: semillas originales, debe reproducir el barrido
mkdir -p $OUT/PPO/logs
echo "$(date +%T) verificacion inicio M=$M"
for md in $MDS; do
  drv=Driver_${md##*_}
  nice -n 10 ~/.venvs/f1-drl-v2/bin/python scripts/gp_configs_fix_20261028.py scripts/ds_rollouts_realhist_20261024.py \
    --gp "${GPN[$md]}" --driver $drv --model-dir $md --algo PPO --M $M --fast-env --legal-mask --real-2024 \
    --sim-future --following-gap-scale 0.26 --base-seed $BASE --prefix-seed 0 --out-dir $OUT/PPO \
    --laps ${=LAPS[$md]} > $OUT/PPO/logs/$md.log 2>&1 &
done
wait
echo "$(date +%T) verificacion fin"
