#!/bin/bash
# 2026-11-02 (noche) — E1 sobre las OCHO elegidas de la seleccion con 10 semillas (e2_all10_20261102): Bahrein s5,
# Saudi s4, Miami s8, Hungria s8, Belgica s10, Paises Bajos s6, Singapur s4, EE.UU. s1. Identico a
# e1_probe_20261102.sh salvo la carpeta de salida (ese sondeo fue sobre las elegidas de 3 semillas y no se pisa).
cd "$(dirname "$0")/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
WRAP="scripts/eq2_noise_20261030.py scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py"
OUT=results/startmix_v10g_20261101/e1_20261102
OUT=results/startmix_v10g_20261101/e1_all10_20261102
mkdir -p "$OUT"
for GP in Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG; do
  $PY $WRAP scripts/start_compound_probe_20261025.py \
      --agents trained_models/retrained_v10g_startmix_20261101/selected --algos PPO --gps $GP --n 200 \
      --out "$OUT/probe_sel_$GP.csv" > "$OUT/probe_sel_$GP.log" 2>&1 &
done
wait
~/.venvs/f1-drl-v2/bin/python - <<'PYEOF'
import glob, pandas as pd
f = sorted(glob.glob('results/startmix_v10g_20261101/e1_all10_20261102/probe_sel_*.csv'))
d = pd.concat([pd.read_csv(p) for p in f], ignore_index=True)
d.to_csv('results/startmix_v10g_20261101/e1_all10_20261102/e1_all10_20261102.csv', index=False)
print(d.to_string(index=False))
PYEOF
