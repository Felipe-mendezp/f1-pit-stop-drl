#!/bin/zsh
# 2026-10-30 — una variante del adelantamiento: run_variant_20261030.sh <tag> <count|fig7> [GP_DRV] [n]
#   tag: lXYZ = gap medio + lambda 0.XYZ del ruido (l000 = v10f vigente, l100 = ruido completo);
#        E = logit en todo gap evaluado con el gap con ruido sin opcion mean (variante E de la ablacion Hungarian)
cd "$(dirname $0)/../.."
source results/real_strategy_check_v10eGQf_20261029/env_20261029.sh
H=results/overtake_noise_frac_20261030
#   sufijo f1 (p. ej. l100f1, Ef1): gap de seguimiento 0.4 + Exp(1) en vez de Exp(0.26)
#   tags extra (results/dispersion_hyp_20261030): opciones separadas por "_", p. ej. l000_r24_emp
#     rNN  = F1_RACE_EFFECT_SD 0.NN (efecto piloto-carrera); emp = gap de seguimiento empirico del GP; empP = empirico
#     de todos los GP; chk = sin cambio (control de equivalencia)
T=$1
for o in ${(s:_:)1}; do case $o in
  r[0-9]*) export F1_RACE_EFFECT_SD=$(printf '%s' "${o#r}" | awk '{printf "%.3f", $1/100}');;
  emp) export F1_FOLLOWING_GAP_EMP=$PWD/results/dispersion_hyp_20261030/blocked_gaps_20261030.csv;;
  x) export F1_RACE_EFFECT_LOGIT=0;;
  empP) export F1_FOLLOWING_GAP_EMP=$PWD/results/dispersion_hyp_20261030/blocked_gaps_20261030.csv F1_FOLLOWING_GAP_EMP_POOLED=1;;
esac; done
set -- ${1%%_*} "${@:2}"
B=${1%f1}
[[ $B != $1 ]] && export F1_FOLLOWING_GAP_SCALE=1.0
case $B in
  E) unset F1_OVERTAKE_MEAN_GAP F1_OVERTAKE_NOISE_FRAC;;
  l*) export F1_OVERTAKE_NOISE_FRAC=$(printf '%s' "${B#l}" | awk '{printf "%.3f", $1/100}');;
esac
mkdir -p $H/logs
if [[ $2 == count ]]; then
  TAG=$T N_EP=${4:-30} $PY scripts/gp_configs_fix_20261028.py $H/ot_count_20261030.py > $H/logs/count_$T.log 2>&1
else
  # sin el envoltorio overtake_v10f (fija MEAN_GAP); las tres opciones vienen de env_20261029.sh salvo lo cambiado arriba
  $PY scripts/gp_configs_fix_20261028.py scripts/validate_real_strategy_20261027.py \
      --scen $2 --gps $3 --n-sims ${4:-500} --out $H/$T > $H/logs/$2_${T}_$3.log 2>&1
fi
echo "$T $2 $3 rc=$? RE=$F1_RACE_EFFECT_SD EMP=${F1_FOLLOWING_GAP_EMP:+1}${F1_FOLLOWING_GAP_EMP_POOLED} MEAN=$F1_OVERTAKE_MEAN_GAP LAM=$F1_OVERTAKE_NOISE_FRAC MAX=$F1_OVERTAKE_MAX_GAP FGS=$F1_FOLLOWING_GAP_SCALE"
