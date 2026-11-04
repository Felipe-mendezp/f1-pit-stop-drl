#!/bin/zsh
# 2026-11-03: Figs. 8-10 del paper en v10g (Saudi NOR, vuelta 7, PPO, escenario (ii) = _real), estilo manuscrito, desde los
# CSV de run_20261103.sh. Copia de ../decision_support_v9r2w24belief_20261023/figs_paper_20261024.sh con el entorno de run_20261103.sh.
# La Fig. 9 vuelve a simular una continuacion por accion (semilla 1000), como en el barrido. Salida: figs_paper_20261103/{pdf,png}/.
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1 OMP_NUM_THREADS=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028 F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_20261030/selected
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
unset F1_RIVAL_DNF F1_START_MIX F1_DS_FORCE
export F1_DS_WORLD2024=1 F1_DS_ALGO=PPO F1_DS_SIM_FUTURE=1 F1_DS_FOLLOWING_GAP_SCALE=0.26 F1_DS_PAPER_STYLE=1 F1_PAPER_FONTS=1
PY=~/.venvs/f1-drl-v2/bin/python
GPN='Saudi Arabian Grand Prix' DRV=Driver_NOR MD=Saudi_Arabian_NOR KEY=saudi_arabian_nor LAP=7
RD=results/decision_support_v10gw24belief_20261103/PPO
O=results/decision_support_v10gw24belief_20261103/figs_paper_20261103
mkdir -p $O/pdf $O/png $O/tmp
$PY scripts/plot_decision_support.py --tag ${KEY}_real_lap$LAP --results-dir $RD \
    --out-pdf $O/pdf/decision_support_real_lap$LAP.pdf --out-png $O/png/decision_support_real_lap$LAP.png
$PY scripts/gp_drivers_v5hd_20261014.py scripts/plot_decision_stints.py --gp "$GPN" \
    --driver $DRV --model-dir $MD --lap $LAP --real-2024 --seed 1000 --results-dir $RD \
    --following-gap-scale 0.26 --sim-future \
    --out-pdf $O/pdf/decision_stints_real_lap$LAP.pdf --out-png $O/png/decision_stints_real_lap$LAP.png
$PY scripts/plot_decision_evolution.py --model-dir ${MD}_real --pdf-dir $O/tmp --png-dir $O/tmp \
    --name ${KEY}_real --drop-last-laps 1 --results-dir $RD
for e in pdf png; do cp $O/tmp/${KEY}_real_evolution_points.$e $O/$e/; done
