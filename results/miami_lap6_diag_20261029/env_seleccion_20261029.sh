# 2026-10-29 diagnostico Miami vuelta 6: mundo de la seleccion de semillas (= entrenamiento). Hacer source desde la raiz del repo.
source reproduce/env_paper_v3_20261007.sh >/dev/null 2>&1
export MPLBACKEND=Agg F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26 OMP_NUM_THREADS=1
export F1_RIVAL_DNF=$PWD/data/rival_dnf_2021_2023_20261014.json
unset F1_WORLD2024_TYRES F1_START_MIX F1_DS_FORCE F1_RL_AGENTS_DIR
PY=~/.venvs/f1-drl-v2/bin/python
WRAP="scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py"
