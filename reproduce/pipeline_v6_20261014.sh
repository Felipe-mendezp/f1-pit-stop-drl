#!/bin/zsh
# Pipeline v6 (2026-10-14): de los modelos entrenados en el cluster (trained_models/rl_agents_v6_20261014/seedXX) a la
# seleccion de semillas, las Figs. 6/7 en el mundo 2024 y el chequeo de paradas tardias. No entrena nada.
#
#   zsh reproduce/pipeline_v6_20261014.sh check          # Saudi + Singapore (tareas 1-100): select, fig7, dnfobs, summary
#   zsh reproduce/pipeline_v6_20261014.sh all            # 8 GP: select, fig6 (abandonos reales), fig7, summary + figuras
#   zsh reproduce/pipeline_v6_20261014.sh <etapa> [GP...] # etapas: select | fig6 | fig7 | dnfobs | summary
#
# Configuracion fija:
#   - fisica del paper v3 (reproduce/env_paper_v3_20261007.sh), simulador trained_models/simulation_v6_20261014
#     (Miami SC/VSC 0.5/0.5), neumaticos del mundo 2024 opcion3h (como la Fig. 6/7 v5h);
#   - pilotos v6 via scripts/gp_drivers_v5hd_20261014.py (Belgian GAS, Hungarian TSU, Dutch STR, US MAG);
#   - seleccion de semillas con abandonos aleatorios de rivales (scripts/rival_dnf_v6_20261014.py), 200 carreras de
#     validacion (semilla 2026), filtro 1-4 paradas y primera despues de la vuelta 4 en >= 95 %;
#   - Fig. 6 con los abandonos reales de 2024 (--real-dnf); Fig. 7 con estrategias reales (ya los incluye);
#     sin F1_RIVAL_DNF en las evaluaciones (con --real-dnf el sorteo no se usa, y asi no se consume RNG).
# Variables opcionales: JOBS (8), NVAL (200), NSIMS (500), TAG (sufijo del arbol de agentes, solo para pruebas),
#   OUT (results/world2024_v6_20261014).
set -e
cd "$(dirname $0)/.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v6_20261014
PY=~/.venvs/f1-drl-v2/bin/python
JOBS=${JOBS:-8}; NVAL=${NVAL:-200}; NSIMS=${NSIMS:-500}; TAG=${TAG:-}
OUT=${OUT:-results/world2024_v6_20261014}
AGENTS=trained_models/rl_agents_v6_20261014$TAG
ALGOS=(DQN A2C TRPO PPO RecurrentPPO)
ALL=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG)
CHECK=(Saudi_Arabian_NOR Singapore_HUL)
mkdir -p $OUT/logs
log() { echo "$(date +%T) $*" | tee -a $OUT/logs/status.txt; }

stage_select() {   # $@ = GP
  log "select eval $*"
  $PY scripts/rival_dnf_v6_20261014.py scripts/select_seeds_v5_20261010.py eval --calib v6 ${TAG:+--tag=$TAG} \
      --gps $@ --algos $ALGOS --n $NVAL --jobs $JOBS > $OUT/logs/select_eval.log 2>&1
  log "select pick $*"
  $PY scripts/rival_dnf_v6_20261014.py scripts/select_seeds_v5_20261010.py select --calib v6 ${TAG:+--tag=$TAG} \
      --gps $@ --algos $ALGOS | tee $OUT/logs/select_pick.log | tail -$(( ${#@} * ${#ALGOS} + 2 ))
}

stage_eval() {     # $1 = fig6 | fig7 | dnfobs ; resto = GP
  local kind=$1; shift
  export F1_RL_AGENTS_DIR=$AGENTS/selected
  test -d $F1_RL_AGENTS_DIR || { echo "falta $F1_RL_AGENTS_DIR (correr la etapa select)"; exit 1; }
  log "$kind $*"
  for g in $@; do echo $g; done | xargs -P $JOBS -L 1 zsh -c '
    g=$0; k='$kind'; o='$OUT'; py='$PY'; n='$NSIMS'
    case $k in
      fig6)   $py scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig 6 --real-dnf \
                --world 2024 --gps $g --n-sims $n --algos '"$ALGOS"' --out $o ;;
      fig7)   $py scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig 7 \
                --world 2024 --gps $g --n-sims $n --algos '"$ALGOS"' --out $o ;;
      dnfobs) $py scripts/gp_drivers_v5hd_20261014.py results/world2024_v5h_ood_20261014/intervene_20261014.py \
                --mode dnfobs --fig 7 --world 2024 --gps $g --n-sims 200 --algos '"$ALGOS"' --out $o/dnfobs ;;
    esac > $o/logs/${k}_$g.log 2>&1
    echo "$(date +%T) $k $g exit $?" >> $o/logs/status.txt'
}

stage_summary() {
  log "summary"
  $PY scripts/gp_drivers_v5hd_20261014.py results/world2024_v6_20261014/summarize_v6_20261014.py --dir $OUT \
      2>&1 | grep -v -i warning | tee $OUT/logs/summary.log
}

case $1 in
  check)   stage_select $CHECK; stage_eval fig7 $CHECK; stage_eval dnfobs $CHECK; stage_summary ;;
  all)     stage_select $ALL; stage_eval fig6 $ALL; stage_eval fig7 $ALL; stage_summary
           [[ -z $TAG ]] && $PY scripts/make_manifest_v6_20261014.py | tee -a $OUT/logs/status.txt ;;
  select)  shift; stage_select ${@:-$ALL} ;;
  fig6|fig7|dnfobs) k=$1; shift; stage_eval $k ${@:-$ALL} ;;
  summary) stage_summary ;;
  *) sed -n 2,20p $0; exit 2 ;;
esac
log "fin $1"
