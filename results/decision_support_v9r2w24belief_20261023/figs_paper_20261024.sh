#!/bin/zsh
# 2026-10-24: Figs. 8-10 del paper v9r2 (Singapore HUL, vuelta 29 = vuelta de la parada real de 2024, PPO, escenario _real)
# en estilo manuscrito, desde los CSV del barrido v9r2 (run_20261023.sh). Copia de
# results/decision_support_v8w24belief_20261019/figs_paper_20261021.sh con el entorno v9r2 (juegos reutilizables).
# La Fig. 9 vuelve a simular una sola continuacion por accion (semilla 1000), como en el barrido.
# Agentes: trained_models/rl_agents_v9r2_20261022/selected (PPO identico al de dsq0/agents/selected).
# Salida: figs_paper_20261024/{pdf,png}/.   LAP=29 por defecto.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7 OMP_NUM_THREADS=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015 F1_RL_AGENTS_DIR=$PWD/trained_models/rl_agents_v9r2_20261022/selected
export F1_DS_WORLD2024=1 F1_DS_ALGO=PPO F1_DS_SIM_FUTURE=1 F1_DS_FOLLOWING_GAP_SCALE=1.0 F1_DS_PAPER_STYLE=1
export F1_PAPER_FONTS=1
PY=~/.venvs/f1-drl-v2/bin/python
LAP=${LAP:-29}
RD=results/decision_support_v9r2w24belief_20261023/PPO
O=results/decision_support_v9r2w24belief_20261023/figs_paper_20261024
mkdir -p $O/pdf $O/png $O/tmp
$PY scripts/plot_decision_support.py --tag singapore_hul_real_lap$LAP --results-dir $RD \
    --out-pdf $O/pdf/decision_support_real_lap$LAP.pdf --out-png $O/png/decision_support_real_lap$LAP.png
$PY scripts/gp_drivers_v5hd_20261014.py scripts/plot_decision_stints.py --gp 'Singapore Grand Prix' \
    --driver Driver_HUL --model-dir Singapore_HUL --lap $LAP --real-2024 --seed 1000 --results-dir $RD \
    --following-gap-scale 1.0 --sim-future \
    --out-pdf $O/pdf/decision_stints_real_lap$LAP.pdf --out-png $O/png/decision_stints_real_lap$LAP.png
$PY scripts/plot_decision_evolution.py --model-dir Singapore_HUL_real --pdf-dir $O/tmp --png-dir $O/tmp \
    --name singapore_hul_real --drop-last-laps 1 --results-dir $RD
for e in pdf png; do cp $O/tmp/singapore_hul_real_evolution_points.$e $O/$e/; done
