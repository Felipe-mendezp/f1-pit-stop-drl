#!/bin/zsh
# 2026-11-11 — Fig. 11 of the paper (fig:realhist_best, realhist_best.pdf): Bahrain GP, Alonso, "DRL - actual" and
# "Best of the four - actual". Same environment as fig11_20261103.sh. The best action is chosen in the sweep
# (compare_laps_20261103.csv) and both columns are measured with the independent rollouts of verif_best_20261111/.
# Output: $OUT/realhist_best_bahrain_alo.{pdf,png} (default figs_best_20261111/).
#   zsh results/decision_support_v10gsmrealhist_20261102/fig11best_20261111.sh
cd "$(dirname $0)/../.."
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v10eGQ F1_VSC_FIRST_LAP_ONLY=1
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_v10eGQ_20261028.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
export F1_TYRE_STINT_AGES=$PWD/data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv
export F1_OVERTAKE_MAX_GAP=inf F1_OVERTAKE_MEAN_GAP=1 F1_FOLLOWING_GAP_SCALE=0.26
export F1_RACE_EFFECT_SD=0.28 F1_RACE_EFFECT_KEEP_STD=1
export F1_DS_FORCE=hist F1_DS_WORLD2024=1 F1_PAPER_FONTS=1
export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v10eGQ_20261028
export F1_RL_AGENTS_DIR=$PWD/trained_models/retrained_v10g_startmix_20261101/selected
unset F1_RIVAL_DNF F1_START_MIX
OUT=${OUT:-results/decision_support_v10gsmrealhist_20261102/figs_best_20261111}
~/.venvs/f1-drl-v2/bin/python scripts/plot_realhist_best_20261111.py --gp bahrain_alo --no-title \
    --res results/decision_support_v10gsmrealhist_20261102 --laps compare_laps_20261103.csv \
    --vals results/decision_support_v10gsmrealhist_20261102/verif_best_20261111 --out $OUT
echo "rc=$?"
