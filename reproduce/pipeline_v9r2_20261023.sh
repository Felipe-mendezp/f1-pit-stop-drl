#!/bin/zsh
# Pipeline v9r2 (2026-10-23): copia de reproduce/pipeline_v8_20261019.sh para los agentes multi-piloto v9r2
# (trained_models/rl_agents_v9r2_20261022 = v8 + juegos de neumaticos 2024 reutilizables, F1_TYRE_SETS + F1_TYRE_REUSE=1,
# ver src/tyre_sets.py) y salida results/world2024_v9r2_20261023. Mismo simulador (v7), mundo 2024 (v7), reglas de
# seleccion 5/7/8 + --max-dsq 0.01 y QMIP con simetria completa (F1_QMIP_SYMMETRY=full; con juegos la QMIP usa
# qmip._solve_sets, que no la lee). Diferencias con v8:
#   - todas las etapas con F1_TYRE_SETS=data/tyre_sets_2024_reg_20261021.csv, F1_TYRE_REUSE=1, F1_NO_SET_PENALTY=-1000
#     (igual que el entrenamiento via scripts/tyre_reuse_v9r2_20261022.py + scripts/tyre_sets_v9_20261021.py);
#   - seleccion con --calib v9r2 (via los mismos envoltorios que el entrenamiento);
#   - etapa nueva qmip: Figs. 6 y 7 solo con la QMIP (directorio de agentes vacio: los DRL se saltan), salida
#     $OUT/qmip_only. No depende de los agentes, se puede correr antes de que termine el entrenamiento;
#   - etapa nueva qmipcheck: las filas QMIP de fig6/fig7 deben ser identicas a las de qmip_only (control);
#   - etapa nueva figs: Figs. 6-7 en 2x4 y Tabla O.1 (scripts en $OUT, sin merge).
#
#   zsh reproduce/pipeline_v9r2_20261023.sh qmip              # ahora (sin agentes)
#   zsh reproduce/pipeline_v9r2_20261023.sh all               # tras traer los modelos: select, fig6, fig7, qmipcheck, summary, figs
#   zsh reproduce/pipeline_v9r2_20261023.sh <etapa> [GP...]   # select | fig6 | fig7 | qmip | qmipcheck | summary | figs
#
# Variables opcionales: JOBS (8), NVAL (200), NSIMS (500), OUT (results/world2024_v9r2_20261023),
#   AGENTS (trained_models/rl_agents_v9r2_20261022), SELOUT (results/seed_selection_v5_20261010/v9r2), MAXDSQ (0.01;
#   variante 23-oct con 0 en results/world2024_v9r2_20261023/dsq0/ = version del paper desde el 24-oct, ver su README). AGENTS y SELOUT
#   distintos solo para pruebas (F1_SELECT_TREE / F1_SELECT_OUT en scripts/select_seeds_v5_20261010.py). OUT y AGENTS
#   relativos a la raiz del repo o absolutos sin espacios (xargs).
set -e -o pipefail
cd "$(dirname $0)/.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000 F1_QMIP_SYMMETRY=full
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF
PY=~/.venvs/f1-drl-v2/bin/python
JOBS=${JOBS:-8}; NVAL=${NVAL:-200}; NSIMS=${NSIMS:-500}
OUT=${OUT:-results/world2024_v9r2_20261023}
AGENTS=${AGENTS:-trained_models/rl_agents_v9r2_20261022}
SELOUT=${SELOUT:-results/seed_selection_v5_20261010/v9r2}
FIGS=results/world2024_v9r2_20261023   # scripts de figuras y tabla (leen $OUT via F1_V9R2_DIR)
ALGOS=(DQN A2C TRPO PPO RecurrentPPO)
ALL=(Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG)
WRAP=(scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py)
mkdir -p $OUT/logs
log() { echo "$(date +%T) $*" | tee -a $OUT/logs/status.txt; }

stage_select() {   # $@ = GP
  export F1_SELECT_TREE=${AGENTS:a} F1_SELECT_OUT=${SELOUT:a}
  test -d $AGENTS/seed01 || { echo "falta $AGENTS/seed01 (traer los modelos del cluster)"; exit 1; }
  log "select eval $* (v9r2: F1_TYRE_SETS=$F1_TYRE_SETS F1_TYRE_REUSE=$F1_TYRE_REUSE)"
  $PY $WRAP scripts/select_seeds_v5_20261010.py eval --calib v9r2 \
      --gps $@ --algos $ALGOS --n $NVAL --jobs $JOBS > $OUT/logs/select_eval.log 2>&1
  log "select pick $*"
  $PY $WRAP scripts/select_seeds_v5_20261010.py select --calib v9r2 \
      --gps $@ --algos $ALGOS --ok-neutral --first-after 5 --late 7 --min-gap 8 --max-dsq ${MAXDSQ:-0.01} | tee $OUT/logs/select_pick.log | tail -$(( ${#@} * ${#ALGOS} + 2 ))
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
      fig6) $py scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig 6 --real-dnf \
              --world 2024 --gps $g --n-sims $n --algos '"$ALGOS"' --out $o ;;
      fig7) $py scripts/gp_drivers_v5hd_20261014.py scripts/evaluate_world2024_20261010.py --fig 7 \
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
  $PY scripts/gp_drivers_v5hd_20261014.py results/world2024_v6_20261014/summarize_v6_20261014.py --dir $OUT \
      2>&1 | grep -v -i warning | tee $OUT/logs/summary.log
}

stage_figs() {
  log "figs"
  F1_V9R2_DIR=${OUT:a} $PY scripts/gp_drivers_v5hd_20261014.py $FIGS/figs67_2x4_20261023.py
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
  *) sed -n 2,22p $0; exit 2 ;;
esac
log "fin $1"
