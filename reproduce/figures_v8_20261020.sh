#!/bin/zsh
# Regenerate the figures and tables of the revised (v8) manuscript from the shipped v8 models and the cached
# simulation results (no training, no long re-evaluation). 2026-10-20.
# The figures are redrawn in place inside results/ and then collected, with the names used by the manuscript's
# \includegraphics, in $OUT (default figures_regenerated_v8/). reproduce/figures.sh (v2.0.0) is left unchanged.
#
#   zsh reproduce/figures_v8_20261020.sh
#   OUT=/tmp/figs zsh reproduce/figures_v8_20261020.sh
#
# Requires trained_models/ with the v8 trees of reproduce/MODELS_MANIFEST_v8_20261020.csv and their simulators
# (simulation_v7_20261015, simulation_onedriver_v7_20261015 and the trees they link to), and gurobipy (Fig. 5 re-solves
# the QMIP). Long runs that produced the cached results: reproduce/pipeline_v8_20261019.sh (Figs. 6-7),
# results/decision_support_v8w24belief_20261019/run_20261019.sh (Figs. 8-10), results/onedriver_v8_20261019/README.md
# (Fig. 5, Table N.1).
set -e
cd "$(dirname $0)/.."
source reproduce/env_paper_v3_20261007.sh >/dev/null
export MPLBACKEND=Agg
PY=${PY:-$HOME/.venvs/f1-drl-v2/bin/python}   # env_paper sets PYTHON=python; use the project venv
OUT=${OUT:-figures_regenerated_v8}
mkdir -p $OUT/main $OUT/supplementary
for d in trained_models/simulation_v7_20261015 trained_models/simulation_onedriver_v7_20261015 \
         trained_models/rl_agents_v8_20261018/selected trained_models/rl_agents_onedriver_v8_20261019; do
  [ -d $d ] || { echo "missing $d (see reproduce/MODELS_MANIFEST_v8_20261020.csv)"; exit 1; }
done
step() { echo; echo "=== $*"; }

step "Fig. 3: compound effects and degradation, United States GP (v7 calibration, 28 laps)"
F1_TYRELIFE_2021=v7 $PY results/sim_v7_20261015/fig3_v7_20261015.py --gp "United States Grand Prix" --laps 28
cp results/sim_v7_20261015/fig3_v7/facet_compounds_United_States.pdf $OUT/main/facet_compounds.pdf

step "Fig. 5 + Table N.1: single-driver benchmark (selected v8 seeds), re-solves the QMIP"
# 2026-10-21: QMIP with full symmetry breaking (F1_QMIP_SYMMETRY=full) and manuscript fonts
bash results/onedriver_v8_20261019/run_fig5_tableN1_qmipsym_20261021.sh
cp results/onedriver_v8_qmipsym_20261021/fig5/figure_5.pdf $OUT/main/stints_alone.pdf
cp results/onedriver_v8_qmipsym_20261021/tableN1/table_M1.tex $OUT/supplementary/tableN1_optimality_gap.tex

step "Figs. 6 and 7: DRL vs QMIP in the 2024 world (from the cached 500-race results)"
zsh reproduce/pipeline_v8_20261019.sh summary
# 2026-10-21: DRL rows of results/world2024_v8_20261019 + QMIP with full symmetry breaking
# (reproduce/qmipsym_figs67_20261021.sh, then merge_20261021.py and the summary/ci_table scripts in merged/)
M=results/world2024_v8_qmipsym_20261021/merged
$PY results/world2024_v8_qmipsym_20261021/merge_20261021.py
(export F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
 $PY scripts/gp_drivers_v5hd_20261014.py results/world2024_v6_20261014/summarize_v6_20261014.py --dir $M > $M/summary.log 2>&1)
$PY $M/ci_table_20261020.py
# 2026-10-21: 2x4 layout in the manuscript
(export F1_WORLD2024_TYRES=v7 F1_SIMULATION_DIR=trained_models/simulation_v7_20261015
 $PY scripts/gp_drivers_v5hd_20261014.py results/world2024_v8_qmipsym_20261021/figs67_2x4_20261021.py > /dev/null 2>&1)
cp $M/fig6_2024_realdnf/figure_6_2x4.pdf $OUT/main/qmip_vs_drl_stochastic_boxplot.pdf
cp $M/fig7_2024/figure_7_2x4.pdf $OUT/main/all_gps_2024_qmip_boxplot.pdf
cp $M/ci_table_20261020.tex $OUT/supplementary/tableO1_ci_2024.tex

step "Figs. 8-10: decision support, Saudi Arabian GP 2024 (NOR), lap 7, PPO (from the cached rollouts)"
# 2026-10-21: manuscript style (--paper-style), see the script header
zsh results/decision_support_v8w24belief_20261019/figs_paper_20261021.sh
F=results/decision_support_v8w24belief_20261019/figs_paper_20261021/pdf
cp $F/decision_support_real_lap7.pdf $F/decision_stints_real_lap7.pdf $F/saudi_arabian_nor_real_evolution_points.pdf $OUT/main/

step "Supplementary tables: rivals' pit-stop and compound logits, overtaking logit (v7 simulator models)"
$PY results/rival_overtake_tables_v7_20261020/make_tables_20261020.py > /dev/null
cp results/rival_overtake_tables_v7_20261020/tab_*.tex $OUT/supplementary/

echo; echo "done -> $OUT/ (compare with the manuscript folder; PDFs embed timestamps, compare visually)"
