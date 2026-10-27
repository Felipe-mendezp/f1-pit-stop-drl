source reproduce/env_paper_v3_20261007.sh >/dev/null 2>&1
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_RL_AGENTS_DIR=trained_models/retrained_v9r2_startmix_20261025/selected OMP_NUM_THREADS=1
unset F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
PY=~/.venvs/f1-drl-v2/bin/python
