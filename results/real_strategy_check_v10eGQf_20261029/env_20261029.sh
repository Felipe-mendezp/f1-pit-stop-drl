# 2026-10-29 — entorno de la validacion con la estrategia real en v10eGQf (= el de reproduce/pipeline_v10_20261027.sh)
source reproduce/env_paper_v3_20261007.sh >/dev/null 2>&1
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RL_AGENTS_DIR=trained_models/retrained_v10eGQf_20261029/selected OMP_NUM_THREADS=1
unset F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
PY=~/.venvs/f1-drl-v2/bin/python
