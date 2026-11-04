#!/bin/bash
# 2026-11-02 - E1 amplia sobre las 5 replicas de Miami_OCO que pasan la regla de seleccion (semillas 3, 5, 7, 8, 10).
# Motivo: la elegida (semilla 8) pasa con la largada real C3 pero descalifica en el 100 % de las carreras si se le
# impone C4. Se mide el DSQ en los cuatro modos para ver si alguna de las que pasan la regla cumple el criterio E1
# original (DSQ <= 1 % bajo cualquier compuesto de largada). Mismo mundo y envoltorios que e1_probe_us_20261102.sh.
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
for S in 03 05 07 08 10; do
  $PY $WRAP scripts/start_compound_probe_20261025.py \
      --agents trained_models/retrained_v10g_startmix_20261101/seed$S --algos PPO --gps Miami_OCO --n 200 \
      --out "$OUT/probe_miami_seed$S.csv" > "$OUT/probe_miami_seed$S.log" 2>&1 &
done
wait
~/.venvs/f1-drl-v2/bin/python - <<'PYEOF'
import glob, pandas as pd
rs = []
for p in sorted(glob.glob('results/startmix_v10g_20261101/e1_20261102/probe_miami_seed*.csv')):
    d = pd.read_csv(p); d['seed'] = int(p[-6:-4]); rs.append(d)
d = pd.concat(rs, ignore_index=True)
d.to_csv('results/startmix_v10g_20261101/e1_20261102/e1_miami_seeds_20261102.csv', index=False)
print(d[['seed','modo','dsq_pct','pos','paradas','largadas']].to_string(index=False))
PYEOF
