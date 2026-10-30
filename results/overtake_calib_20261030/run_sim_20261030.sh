#!/bin/zsh
# 2026-10-30 — sim_pairs para un tag de variante (mismas opciones que ../overtake_noise_frac_20261030/run_variant), 8 GP en 4 procesos
cd "$(dirname $0)/../.."
source results/real_strategy_check_v10eGQf_20261029/env_20261029.sh
T=$1; N=${2:-100}
for o in ${(s:_:)T}; do case $o in
  r[0-9]*) export F1_RACE_EFFECT_SD=$(printf '%s' "${o#r}" | awk '{printf "%.3f", $1/100}');;
  k) export F1_RACE_EFFECT_KEEP_STD=1;;
  fg[0-9]*) export F1_FOLLOWING_GAP_SCALE=$(printf '%s' "${o#fg}" | awk '{printf "%.3f", $1/100}');;
esac; done
export F1_OVERTAKE_NOISE_FRAC=0.000
mkdir -p results/overtake_calib_20261030/logs
for g in Bahrain Belgian Dutch Hungarian Miami "Saudi Arabian" Singapore "United States"; do echo "$g"; done | \
  xargs -P 4 -I{} sh -c "TAG=$T GPS='{}' N_EP=$N $PY scripts/gp_configs_fix_20261028.py results/overtake_calib_20261030/sim_pairs_20261030.py > 'results/overtake_calib_20261030/logs/${T}_{}.log' 2>&1"
echo "$T listo RE=$F1_RACE_EFFECT_SD KEEP=$F1_RACE_EFFECT_KEEP_STD FGS=$F1_FOLLOWING_GAP_SCALE MEAN=$F1_OVERTAKE_MEAN_GAP MAX=$F1_OVERTAKE_MAX_GAP"
