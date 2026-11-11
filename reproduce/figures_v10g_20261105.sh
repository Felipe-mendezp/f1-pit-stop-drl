#!/bin/zsh
# Regenerate the figures and tables of the manuscript (v2.0.0, simulator v10eGQ, agents v10g and v10gsm) from the shipped
# models and the cached results: no training and no long re-evaluation. 2026-11-05.
# The figures are redrawn in place inside results/ and collected, with the names used by the manuscript's
# \includegraphics, in $OUT (default figures_regenerated_v2.0.0/). Figs. 1, 2 and 4 are drawn in LaTeX.
#
#   zsh reproduce/figures_v10g_20261105.sh
#   OUT=/tmp/figs zsh reproduce/figures_v10g_20261105.sh
#
# Requires at the repository root the two Zenodo archives, f1-pit-stop-drl-v2.0.0-models.tar.gz and
# f1-pit-stop-drl-v2.0.0-rollouts.tar.gz, gurobipy (Fig. 5 re-solves the QMIP) and the environment in
# ~/.venvs/f1-drl-v2, the interpreter the results/ scripts call.
# Long runs behind the cached results: reproduce/pipeline_v10g_20261031.sh (Figs. 6-7),
# results/decision_support_v10gw24belief_20261103/run_20261103.sh (Figs. 8-10),
# results/decision_support_v10gsmrealhist_20261102/run_20261102.sh (Fig. 11 and Fig. Q.1) and
# results/decision_support_v10gsmrealhist_20261102/verif_best_20261111/run_verif_20261111.sh (Fig. 11, about 4 min).
# 2026-11-11: Fig. 11 is now the Bahrain figure (realhist_best.pdf); the 8-GP figure moved to the supplementary
# material as Fig. Q.1; Table P.1 carries significance marks.
set -e
cd "$(dirname $0)/.."
PY=${PY:-$HOME/.venvs/f1-drl-v2/bin/python}
OUT=${OUT:-figures_regenerated_v2.0.0}
typeset +x OUT   # 2026-11-11: keep it out of the environment; reproduce/pipeline_v10g_20261031.sh reads its own $OUT
DS=results/decision_support_v10gw24belief_20261103
RH=results/decision_support_v10gsmrealhist_20261102
W=results/world2024_v10g_20261031
SIM=results/sim_v10eGQ_20261028
mkdir -p $OUT/main $OUT/supplementary
for d in trained_models/simulation_v10eGQ_20261028 trained_models/simulation_onedriver_v10eGQ_20261028 \
         trained_models/retrained_v10g_20261030/selected trained_models/retrained_v10g_startmix_20261101/selected \
         $DS/PPO $RH/PPO $RH/verif_best_20261111/PPO; do
  [ -d $d ] || { echo "missing $d (extract the Zenodo archives at the repository root)"; exit 1; }
done
step() { echo; echo "=== $*"; }

step "Fig. 3 and Fig. H.1: compound effects and outlier laps, United States GP"
source reproduce/env_paper_v3_20261007.sh >/dev/null
export MPLBACKEND=Agg
F1_TYRELIFE_2021=v10eGQ $PY $SIM/fig3_v10_20261027.py --laps 28
cp $SIM/fig3_v10/facet_compounds_United_States.pdf $OUT/main/facet_compounds.pdf
F1_TYRELIFE_2021=v10eGQ $PY $SIM/outliers_us_v10_20261027.py
cp $SIM/outliers_us_v10/laptimes_outliers_us.pdf $OUT/supplementary/laptimes_outliers_2022.pdf

step "Fig. 5 and Table N.1: single-driver benchmark, re-solves the QMIP"
bash results/onedriver_v10eGQ_20261028/run_fig5_tableN1_20261028.sh
cp results/onedriver_v10eGQ_20261028/fig5/figure_5.pdf $OUT/main/stints_alone.pdf
cp results/onedriver_v10eGQ_20261028/tableN1/table_M1.tex $OUT/supplementary/tableN1_optimality_gap.tex

step "Figs. 6 and 7 and Table P.1: DRL vs QMIP, from the cached 500-race results"
zsh reproduce/pipeline_v10g_20261031.sh figs
(export F1_WORLD2024_TYRES=v10eGQ F1_SIMULATION_DIR=trained_models/simulation_v10eGQ_20261028
 F1_V9R2_DIR=$PWD/$W F1_REALSTRAT_DIR=$PWD/results/overtake_noise_frac_20261030/l000_r28_k \
   $PY scripts/gp_configs_fix_20261028.py results/world2024_v9r2_20261023/fig7_actual_20261027.py)
cp $W/fig6_2024_realdnf/figure_6_2x4.pdf $OUT/main/qmip_vs_drl_stochastic_boxplot.pdf
cp $W/fig7_2024/figure_7_2x4_actual.pdf $OUT/main/all_gps_2024_qmip_boxplot.pdf
$PY $W/ci_table_v10g_20261111.py
cp $W/ci_table_v10g_20261111.tex $OUT/supplementary/tableP1_ci_2024.tex

step "Figs. 8-10: look-ahead, Saudi Arabian GP 2024 (NOR), lap 7, PPO, from the cached rollouts"
zsh $DS/figs_paper_20261103.sh
F=$DS/figs_paper_20261103/pdf
cp $F/decision_support_real_lap7.pdf $F/decision_stints_real_lap7.pdf $F/saudi_arabian_nor_real_evolution_points.pdf $OUT/main/

step "Fig. 11, Fig. Q.1 and Table L.1: review of the 2024 decisions, from the cached rollouts"
OUT=$OUT/tmp_fig11 zsh $RH/fig11best_20261111.sh
cp $OUT/tmp_fig11/realhist_best_bahrain_alo.pdf $OUT/main/realhist_best.pdf
rm -rf $OUT/tmp_fig11
zsh $RH/fig11_20261103.sh
cp $RH/figs/realhist_stints_grid4x2.pdf $OUT/supplementary/realhist_stints.pdf
zsh $RH/tyre_sets_20261103/run_20261103.sh
$PY scripts/table_tyre_sets_decisions_20261103.py --res $RH --laps compare_laps_20261103.csv \
    --out $OUT/supplementary/tab_tyre_sets_decisions.tex

echo; echo "done -> $OUT/ (compare with the manuscript's figures; PDFs embed timestamps, compare visually)"
