#!/bin/zsh
# 2026-10-21: Figs. 8-10 del paper (Saudi NOR, vuelta 7, PPO, escenario _real) en estilo manuscrito (--paper-style),
# desde los CSV guardados del barrido v8 (run_20261019.sh). Mismo entorno que el paso --figures del barrido
# (run_ds_v6_sweep_20261015.py: figures() y post_real_figures.py); la Fig. 9 vuelve a simular una sola continuacion
# por accion (semilla 1000, eventos reales 2024), como en el barrido. Salida: figs_paper_20261021/{pdf,png}/.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 OMP_NUM_THREADS=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
unset F1_RIVAL_DNF
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015 F1_RL_AGENTS_DIR=$PWD/trained_models/rl_agents_v8_20261018/selected
export F1_DS_WORLD2024=1 F1_DS_ALGO=PPO F1_DS_SIM_FUTURE=1 F1_DS_FOLLOWING_GAP_SCALE=1.0 F1_DS_PAPER_STYLE=1
export F1_PAPER_FONTS=1  # 2026-10-21: letras 10 % mas chicas y ticks del tamano de la leyenda (scripts/paper_fonts_20261021.py)
PY=~/.venvs/f1-drl-v2/bin/python
RD=results/decision_support_v8w24belief_20261019/PPO
O=results/decision_support_v8w24belief_20261019/figs_paper_20261021
mkdir -p $O/pdf $O/png $O/tmp
$PY scripts/plot_decision_support.py --tag saudi_arabian_nor_real_lap7 --results-dir $RD --ymax 0.29 \
    --out-pdf $O/pdf/decision_support_real_lap7.pdf --out-png $O/png/decision_support_real_lap7.png
$PY scripts/gp_drivers_v5hd_20261014.py scripts/plot_decision_stints.py --gp 'Saudi Arabian Grand Prix' \
    --driver Driver_NOR --model-dir Saudi_Arabian_NOR --lap 7 --real-2024 --seed 1000 --results-dir $RD \
    --following-gap-scale 1.0 --sim-future \
    --out-pdf $O/pdf/decision_stints_real_lap7.pdf --out-png $O/png/decision_stints_real_lap7.png
$PY scripts/plot_decision_evolution.py --model-dir Saudi_Arabian_NOR_real --pdf-dir $O/tmp --png-dir $O/tmp \
    --name saudi_arabian_nor_real --drop-last-laps 1 --results-dir $RD
for e in pdf png; do cp $O/tmp/saudi_arabian_nor_real_evolution_points.$e $O/$e/; done
