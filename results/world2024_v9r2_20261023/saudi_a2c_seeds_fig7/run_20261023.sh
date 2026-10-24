#!/bin/zsh
# Diagnostico: Fig. 7 Saudi A2C con cada una de las 10 semillas v9r2 (mismo entorno que pipeline_v9r2_20261023.sh fig7)
cd "/Users/charlesthraves/MIT Dropbox/Charles Thraves/Research/F1 DLR/Codigos/f1-pit-stop-drl-v2"
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF
T=$1
one() {
  s=$1
  mkdir -p $T/agents$s/Saudi_Arabian_NOR $T/out$s/logs
  ln -sfn "$PWD/trained_models/rl_agents_v9r2_20261022/seed$s/Saudi_Arabian_NOR/A2C" $T/agents$s/Saudi_Arabian_NOR/A2C
  F1_RL_AGENTS_DIR=$T/agents$s nice -n 5 ~/.venvs/f1-drl-v2/bin/python scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py \
    --fig 7 --world 2024 --gps Saudi_Arabian_NOR --n-sims 500 --algos A2C --out $T/out$s > $T/out$s/logs/run.log 2>&1
  echo "$(date +%T) seed$s exit $?"
}
for b in "01 02 03" "04 05 06" "07 08 09" "10"; do
  for s in ${=b}; do one $s & done; wait
done
