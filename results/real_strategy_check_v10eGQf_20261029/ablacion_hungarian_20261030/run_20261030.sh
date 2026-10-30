#!/bin/bash
# $1 variant $2 gp
cd "/Users/charlesthraves/MIT Dropbox/Charles Thraves/Research/F1 DLR/Codigos/f1-pit-stop-drl-v2"
source results/real_strategy_check_v10eGQf_20261029/env_20261029.sh
OUT=$CLAUDE_JOB_DIR/tmp/hun/$1
V=$1; GP=${2:-Hungarian_TSU}
case $V in
 A) W="scripts/overtake_v10f_20261029.py";;
 D) W=""; export F1_FOLLOWING_GAP_SCALE=1.0;;   # v10f logit+meangap, old following gap
 B) W=""; unset F1_OVERTAKE_MAX_GAP F1_OVERTAKE_MEAN_GAP; export F1_FOLLOWING_GAP_SCALE=1.0;;  # new tree, old physics
 C) W=""; unset F1_OVERTAKE_MAX_GAP F1_OVERTAKE_MEAN_GAP; export F1_FOLLOWING_GAP_SCALE=1.0 F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028_old_preadelantamiento;;
 E) W=""; unset F1_OVERTAKE_MEAN_GAP; export F1_FOLLOWING_GAP_SCALE=1.0;;
 F) W=""; unset F1_OVERTAKE_MAX_GAP; export F1_FOLLOWING_GAP_SCALE=1.0;;
esac
$PY $W scripts/gp_configs_fix_20261028.py scripts/validate_real_strategy_20261027.py --scen fig7 --gps $GP --n-sims 300 --out $OUT > $CLAUDE_JOB_DIR/tmp/hun/$V_$GP.log 2>&1
echo "$V $GP rc=$? FGS=$F1_FOLLOWING_GAP_SCALE TREE=$F1_SIMULATION_DIR MAX=$F1_OVERTAKE_MAX_GAP"
