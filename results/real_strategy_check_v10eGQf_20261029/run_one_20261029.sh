#!/bin/zsh
# 2026-10-29 — una corrida de la validacion con la estrategia real (v10eGQf): run_one_20261029.sh <fig6|fig7> <GP_DRV> [n]
cd "$(dirname $0)/../.."
source results/real_strategy_check_v10eGQf_20261029/env_20261029.sh
$PY scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/validate_real_strategy_20261027.py \
    --scen $1 --gps $2 --n-sims ${3:-500} --out results/real_strategy_check_v10eGQf_20261029 \
    > results/real_strategy_check_v10eGQf_20261029/logs/$1_$2.log 2>&1
echo "$1 $2 rc=$?"
