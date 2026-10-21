#!/bin/zsh
# 2026-10-21: QMIP de las Figs. 6 y 7 v8 con simetria completa (F1_QMIP_SYMMETRY=full, src/optimization/qmip.py).
# Mismo entorno y comandos que reproduce/pipeline_v8_20261019.sh (etapas fig6/fig7), con --algos PPO (el evaluador
# exige un metodo DRL; sirve de control: sus filas deben ser identicas a las de results/world2024_v8_20261019) y
# salida results/world2024_v8_qmipsym_20261021. Despues: results/world2024_v8_qmipsym_20261021/merge_20261021.py.
cd "$(dirname $0)/.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_RL_AGENTS_DIR=trained_models/rl_agents_v8_20261018/selected
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
OUT=results/world2024_v8_qmipsym_20261021
ALL=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG)
mkdir -p $OUT/logs
for k in 6 7; do
  echo "$(date +%T) fig$k start" >> $OUT/logs/status.txt
  for g in $ALL; do
    extra=(); [[ $k == 6 ]] && extra=(--real-dnf)
    ( $PY scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig $k $extra \
        --world 2024 --gps $g --n-sims 500 --algos PPO --out $OUT > $OUT/logs/fig${k}_$g.log 2>&1
      echo "$(date +%T) fig$k $g exit $?" >> $OUT/logs/status.txt ) &
  done
  wait
done
echo "$(date +%T) fin" >> $OUT/logs/status.txt
