#!/bin/zsh
# Pipeline v10 (2026-10-27): copia de reproduce/pipeline_v9r2_20261023.sh para el simulador v10
# 2026-10-28: scripts/gp_configs_fix_20261028.py (v5hd + estrategias reales de los rivales corregidas) en lugar de
# gp_drivers_v5hd_20261014.py y al inicio de WRAP.
# 2026-10-28 (tarde): arbol y agentes v10eGQ: simulation_v10eGQ_20261028 / retrained_v10eGQ_20261028 (edad efectiva GQ,
# results/tyrelife_effective_20261028/README.md; edades de los juegos 2024 en vueltas de empuje via F1_TYRE_SETS y
# F1_TYRE_STINT_AGES; mundo 2024 = F1_WORLD2024_TYRES=v10eGQ). Antes, 28-oct: arbol y agentes v10d: simulation_v10d_20261028 / retrained_v10d_20261028 (compuestos de FastF1 corregidos con OpenF1 y el live timing, pilotos re-estimados; mundo 2024 = F1_WORLD2024_TYRES=v10d). Los arreglos del 27 y del 28-oct (v10) se cancelaron.
# (trained_models/simulation_v10d_20261028: paradas reales en vez de PitIn de FastF1, TyreLife 2021 x 0.740, VSC solo en
# la 1a vuelta para agente y QMIP; results/sim_v10_20261027/README.md) y los agentes trained_models/retrained_v10d_20261028.
# Diferencias con v9r2: mundo 2024 = tyre_v10 (F1_WORLD2024_TYRES=v10), F1_VSC_FIRST_LAP_ONLY=1 en todas las etapas (y el
# envoltorio scripts/v10_wrapper_20261027.py), --calib v10, salida results/world2024_v10d_20261028, y --max-dsq 0 por
# defecto (la variante dsq0 es la version del paper desde el 24-oct). Las etapas figs usan los scripts de
# results/world2024_v9r2_20261023 (leen $OUT via F1_V9R2_DIR).
# 2026-10-29: adelantamiento v10f (results/sim_v10eGQ_20261028/README_adelantamiento_20261029.md): agentes
# retrained_v10eGQf_20261029, salida results/world2024_v10eGQf_20261029 y seleccion en .../v10f. Las variables de
# scripts/overtake_v10f_20261029.py (F1_OVERTAKE_MAX_GAP=inf, F1_OVERTAKE_MEAN_GAP=1, F1_FOLLOWING_GAP_SCALE=0.26) se
# exportan aqui (pisan el 1.0 de env_paper.sh) y el envoltorio va ademas primero en WRAP y en las etapas fig6/fig7.
#   zsh reproduce/pipeline_v10_20261027.sh qmip              # sin agentes
#   zsh reproduce/pipeline_v10_20261027.sh all               # tras traer los modelos
#   zsh reproduce/pipeline_v10_20261027.sh <etapa> [GP...]   # select | fig6 | fig7 | qmip | qmipcheck | summary | figs
#   variante 23-oct con 0 en results/world2024_v9r2_20261023/dsq0/ = version del paper desde el 24-oct, ver su README). AGENTS y SELOUT
#   distintos solo para pruebas (F1_SELECT_TREE / F1_SELECT_OUT en scripts/select_seeds_v5_20261010.py). OUT y AGENTS
#   relativos a la raiz del repo o absolutos sin espacios (xargs).
set -e -o pipefail
cd "$(dirname $0)/.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028 F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26   # 2026-10-29: v10f
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
JOBS=${JOBS:-8}; NVAL=${NVAL:-200}; NSIMS=${NSIMS:-500}
OUT=${OUT:-results/world2024_v10eGQf_20261029}
AGENTS=${AGENTS:-trained_models/retrained_v10eGQf_20261029}
SELOUT=${SELOUT:-results/seed_selection_v5_20261010/v10f}
FIGS=results/world2024_v9r2_20261023   # scripts de figuras y tabla (leen $OUT via F1_V9R2_DIR)
ALGOS=(DQN A2C TRPO PPO RecurrentPPO)
ALL=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG)
WRAP=(scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py)
mkdir -p $OUT/logs
log() { echo "$(date +%T) $*" | tee -a $OUT/logs/status.txt; }

stage_select() {   # $@ = GP
  export F1_SELECT_TREE=${AGENTS:a} F1_SELECT_OUT=${SELOUT:a}
  test -d $AGENTS/seed01 || { echo "falta $AGENTS/seed01 (traer los modelos del cluster)"; exit 1; }
  log "select eval $* (v10f: F1_OVERTAKE_MAX_GAP=$F1_OVERTAKE_MAX_GAP F1_FOLLOWING_GAP_SCALE=$F1_FOLLOWING_GAP_SCALE F1_TYRE_SETS=$F1_TYRE_SETS F1_TYRE_STINT_AGES=$F1_TYRE_STINT_AGES F1_TYRE_REUSE=$F1_TYRE_REUSE)"
  $PY $WRAP scripts/select_seeds_v5_20261010.py eval --calib v10 \
      --gps $@ --algos $ALGOS --n $NVAL --jobs $JOBS > $OUT/logs/select_eval.log 2>&1
  log "select pick $*"
  $PY $WRAP scripts/select_seeds_v5_20261010.py select --calib v10 \
      --gps $@ --algos $ALGOS --ok-neutral --first-after 5 --late 7 --min-gap 8 --max-dsq ${MAXDSQ:-0} | tee $OUT/logs/select_pick.log | tail -$(( ${#@} * ${#ALGOS} + 2 ))
}

stage_eval() {     # $1 = fig6 | fig7 ; $2 = carpeta de agentes ; $3 = salida ; resto = GP
  local kind=$1 agents=$2 out=$3; shift 3
  export F1_RL_AGENTS_DIR=$agents
  test -d $F1_RL_AGENTS_DIR || { echo "falta $F1_RL_AGENTS_DIR (correr la etapa select)"; exit 1; }
  mkdir -p $out/logs
  log "$kind $* (agentes $agents -> $out)"
  for g in $@; do echo $g; done | xargs -P $JOBS -L 1 zsh -c '
    g=$0; k='$kind'; o='$out'; py='$PY'; n='$NSIMS'
    case $k in
      fig6) $py scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/evaluate_world2024_20261010.py --fig 6 --real-dnf \
              --world 2024 --gps $g --n-sims $n --algos '"$ALGOS"' --out $o ;;
      fig7) $py scripts/overtake_v10f_20261029.py scripts/gp_configs_fix_20261028.py scripts/evaluate_world2024_20261010.py --fig 7 \
              --world 2024 --gps $g --n-sims $n --algos '"$ALGOS"' --out $o ;;
    esac > $o/logs/${k}_$g.log 2>&1
    echo "$(date +%T) $k $g exit $?" >> $o/logs/status.txt'
}

stage_qmipcheck() {
  log "qmipcheck"
  $PY - $OUT <<'EOF'
import glob, os, sys
import pandas as pd
out = sys.argv[1]
bad = 0
for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    for f in sorted(glob.glob(os.path.join(out, 'qmip_only', fig, '*.csv'))):
        g = os.path.join(out, fig, os.path.basename(f))
        if not os.path.exists(g):
            print('sin corrida completa:', g); continue
        a, b = (pd.read_csv(x) for x in (f, g))
        a, b = (d[d.algo == 'QMIP'].reset_index(drop=True) for d in (a, b))
        same = (len(a) == len(b) and a.final_position.fillna(-1).eq(b.final_position.fillna(-1)).all()
                and (a.pit_laps.fillna('').astype(str) == b.pit_laps.fillna('').astype(str)).all())
        bad += not same
        print(f'{fig:18s} {os.path.basename(f)[:-4]:20s} QMIP identica: {same}')
sys.exit(1 if bad else 0)
EOF
}

stage_summary() {
  log "summary"
  $PY scripts/gp_configs_fix_20261028.py results/world2024_v6_20261014/summarize_v6_20261014.py --dir $OUT \
      2>&1 | grep -v -i warning | tee $OUT/logs/summary.log
}

stage_figs() {
  log "figs"
  F1_V9R2_DIR=${OUT:a} $PY scripts/gp_configs_fix_20261028.py $FIGS/figs67_2x4_20261023.py
  F1_V9R2_DIR=${OUT:a} $PY $FIGS/ci_table_v9r2_20261023.py
}

case $1 in
  all)       stage_select $ALL; stage_eval fig6 $AGENTS/selected $OUT $ALL; stage_eval fig7 $AGENTS/selected $OUT $ALL
             stage_qmipcheck; stage_summary; stage_figs ;;
  select)    shift; stage_select ${@:-$ALL} ;;
  fig6|fig7) k=$1; shift; stage_eval $k $AGENTS/selected $OUT ${@:-$ALL} ;;
  qmip)      shift; mkdir -p $OUT/qmip_only/no_agents
             stage_eval fig6 $OUT/qmip_only/no_agents $OUT/qmip_only ${@:-$ALL}
             stage_eval fig7 $OUT/qmip_only/no_agents $OUT/qmip_only ${@:-$ALL} ;;
  qmipcheck) stage_qmipcheck ;;
  summary)   stage_summary ;;
  figs)      stage_figs ;;
  *) sed -n 2,12p $0; exit 2 ;;
esac
log "fin $1"
