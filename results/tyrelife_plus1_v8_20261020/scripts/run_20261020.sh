#!/bin/zsh
cd "/Users/charlesthraves/MIT Dropbox/Charles Thraves/Research/F1 DLR/Codigos/f1-pit-stop-drl-v2"
source reproduce/env_paper_v3_20261007.sh >/dev/null
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015 F1_RL_AGENTS_DIR=trained_models/rl_agents_v8_20261018/selected OMP_NUM_THREADS=1 F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_TL_OFFSET=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
unset F1_RIVAL_DNF
O=results/tyrelife_plus1_v8_20261020; mkdir -p $O/logs
for fig in 6 7; do
  extra=(); [[ $fig == 6 ]] && extra=(--real-dnf)
  for g in Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG; do
    nice -n 10 ~/.venvs/f1-drl-v2/bin/python results/tyrelife_plus1_v8_20261020/scripts/tl_plus1_20261020.py scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig $fig $extra --world 2024 --gps $g --n-sims 500 --algos PPO RecurrentPPO --out $O > $O/logs/fig${fig}_$g.log 2>&1 &
  done
  wait
  echo "$(date +%T) fig$fig done" >> $O/logs/status.txt
done
