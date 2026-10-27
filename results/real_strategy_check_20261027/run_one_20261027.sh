#!/bin/zsh
# 2026-10-27 — una corrida de la validacion con la estrategia real: run_one_20261027.sh <fig6|fig7> <GP_DRV> [n]
cd "$(dirname $0)/../.."
source results/real_strategy_check_20261027/env_20261027.sh
$PY scripts/gp_drivers_v5hd_20261014.py scripts/validate_real_strategy_20261027.py --scen $1 --gps $2 --n-sims ${3:-500} \
    > results/real_strategy_check_20261027/logs/$1_$2.log 2>&1
echo "$1 $2 rc=$?"
