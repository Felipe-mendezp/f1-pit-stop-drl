#!/bin/zsh
# 2026-11-02 (noche) — Barrido B (historia real) para la Fig. 11 con los agentes post-entrenados v10gsm:
# trained_models/retrained_v10g_startmix_20261101/selected (10 semillas por carrera, regla del paper mas el control E1
# con la largada real; ver results/startmix_v10g_20261101/README.md). Reemplaza al barrido v9r2 del 26-oct
# (results/decision_support_v9r2smrealhist_20261026), que es el que todavia alimenta la Fig. 11 del manuscrito.
# Mundo v10g: ruido de la Ec. 2 por variable de entorno (equivalente a scripts/eq2_noise_20261030.py, que solo exporta
# F1_RACE_EFFECT_SD y F1_RACE_EFFECT_KEEP_STD), adelantamiento v10f, edad efectiva v10eGQ y VSC solo en la 1a vuelta.
# Solo PPO: el manuscrito dejo RPPO fuera de la §7 y del apendice el 27-oct. M = 500, vueltas 1..N-1, 16 corridas.
#   zsh results/decision_support_v10gsmrealhist_20261102/run_20261102.sh
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
export F1_DS_FORCE=hist
unset F1_RIVAL_DNF F1_START_MIX
if [[ -n "$PRUEBA" ]]; then
  ~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v10gsmhist --algos PPO \
      --gps Bahrain_ALO --M 2 --workers 1
  exit $?
fi
echo "$(date +%T) barrido B v10gsm inicio"
~/.venvs/f1-drl-v2/bin/python scripts/run_ds_v6_sweep_20261015.py --version v10gsmhist --algos PPO \
    --M 500 --workers 8 --figures
echo "$(date +%T) barrido B v10gsm fin rc=$?"
