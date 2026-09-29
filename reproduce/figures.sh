#!/usr/bin/env bash
# Regenerate every final figure and table of the paper from the shipped trained models and
# cached simulation results (minutes; no training, no long re-evaluation).
# Requires the Zenodo archive extracted into trained_models/ (see trained_models/README.md)
# and a working gurobipy (the QMIP benchmark of Fig. 5 is re-solved).
#
#   bash reproduce/figures.sh            # -> figures_regenerated/{main,supplementary}
#   OUT=/tmp/figs bash reproduce/figures.sh
#
# The published versions live in figures/; at the end, tables and CSVs are compared byte-by-byte.
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env_paper.sh
OUT="${OUT:-figures_regenerated}"
mkdir -p "$OUT/main" "$OUT/supplementary" "$OUT/tmp"
for d in trained_models/simulation trained_models/rl_agents trained_models/rl_agents_reward; do
  [ -d "$d" ] || { echo "missing $d — extract the Zenodo archive first (trained_models/README.md)"; exit 1; }
done
step() { echo; echo "=== $*"; }

step "Fig. 3 — compound effects and degradation, Belgian GP"
$PYTHON scripts/plot_fig3_facet_compounds.py --gp "Belgian Grand Prix" --output "$OUT/main/fig03_facet_compounds_belgian.pdf"

step "Fig. 4 — driver-coefficient prediction error"
$PYTHON scripts/plot_driver_coef_mse.py --output "$OUT/main/fig04_driver_coef_mse.png"

step "Fig. 5 + Table N.1 — single-driver benchmark (5 % compound filter), re-solves the QMIP"
$PYTHON scripts/one_driver_filter_20260925.py fig5 --output "$OUT/main/fig05_stints_single_driver.png"
$PYTHON scripts/one_driver_filter_20260925.py m1 --out-dir "$OUT/tmp/n1"
cp "$OUT/tmp/n1/table_M1.tex" "$OUT/supplementary/tableN1_optimality_gap.tex"
cp "$OUT/tmp/n1/table_M1.csv" "$OUT/tmp/tableN1_optimality_gap.csv"

step "Figs. 6 and 7 — DRL vs QMIP (from the cached 500-race results)"
$PYTHON scripts/evaluate_qmip_vs_drl.py --plot-only
$PYTHON scripts/evaluate_all_gps_with_qmip.py --plot-only
T=trained_models/rl_agents
for e in pdf png; do
  cp "$T/qmip_vs_drl_stochastic_boxplot.$e" "$OUT/main/fig06_qmip_vs_drl_stochastic.$e"
  cp "$T/all_gps_2024_qmip_boxplot.$e" "$OUT/main/fig07_real_2024_replay.$e"
done
cp "$T/qmip_vs_drl_stochastic_table.tex" "$OUT/main/table_fig06_qmip_vs_drl_stochastic.tex"
cp "$T/all_gps_2024_qmip_table.tex" "$OUT/main/table_fig07_real_2024_replay.tex"

step "Figs. 8-10 — decision support, Singapore GP 2024 (HUL), PPO agent"
RD=results/decision_support
F1_DS_ALGO=PPO F1_DS_PAPER_STYLE=1 $PYTHON scripts/plot_decision_support.py --tag singapore_hul_real_lap18 \
    --results-dir "$RD" --out-pdf "$OUT/main/fig08_decision_support_singapore_lap18.pdf" \
    --out-png "$OUT/main/fig08_decision_support_singapore_lap18.png"
F1_DS_ALGO=PPO F1_DS_PAPER_STYLE=1 $PYTHON scripts/plot_decision_stints.py --gp 'Singapore Grand Prix' \
    --driver Driver_HUL --model-dir Singapore_HUL --lap 18 --real-2024 --seed 1000 --results-dir "$RD" \
    --following-gap-scale 1.0 --sim-future --algo PPO \
    --out-pdf "$OUT/main/fig09_decision_stints_singapore_lap18.pdf" \
    --out-png "$OUT/main/fig09_decision_stints_singapore_lap18.png"
F1_DS_ALGO=PPO F1_DS_PAPER_STYLE=1 $PYTHON scripts/plot_decision_evolution.py --model-dir Singapore_HUL_real \
    --pdf-dir "$OUT/tmp/ds" --png-dir "$OUT/tmp/ds" --name singapore_hul_real --drop-last-laps 1 --results-dir "$RD"
for e in pdf png; do
  cp "$OUT/tmp/ds/singapore_hul_real_evolution_points.$e" "$OUT/main/fig10_decision_evolution_singapore.$e"
done

step "Fig. H.1 — lap-time outliers"
$PYTHON scripts/plot_fig_h1_outlier_laptimes.py --output "$OUT/supplementary/laptimes_outliers_2022.png"

step "Section P — reward-function comparison (from the cached results)"
W=trained_models/rl_agents_reward
export F1_RL_AGENTS_DIR="$W" F1_RL_REWARD_DIR="$W"
$PYTHON scripts/evaluate_reward_comparison.py --plot-only
$PYTHON scripts/evaluate_reward_comparison_stochastic.py --plot-only
$PYTHON scripts/reward_comparison_stats.py
$PYTHON scripts/reward_comparison_stats.py --stochastic
for f in reward_comparison_boxplot reward_comparison_stats reward_comparison_stochastic_boxplot reward_comparison_stochastic_stats; do
  for e in pdf png; do cp "$W/$f.$e" "$OUT/supplementary/sectionP_$f.$e"; done
done
cp "$W"/reward_comparison*_pvalues.csv "$OUT/tmp/"
unset F1_RL_AGENTS_DIR F1_RL_REWARD_DIR

step "Comparison with the published tables / CSVs (figures/, results/)"
$PYTHON - "$OUT" <<'PY'
import filecmp, os, sys
out = sys.argv[1]
pairs = [
    ('figures/main/table_fig06_qmip_vs_drl_stochastic.tex', f'{out}/main/table_fig06_qmip_vs_drl_stochastic.tex'),
    ('figures/main/table_fig07_real_2024_replay.tex', f'{out}/main/table_fig07_real_2024_replay.tex'),
    ('figures/supplementary/tableN1_optimality_gap.tex', f'{out}/supplementary/tableN1_optimality_gap.tex'),
    ('results/fig05_single_driver/tableN1_optimality_gap.csv', f'{out}/tmp/tableN1_optimality_gap.csv'),
    ('results/sectionP_reward_comparison/reward_comparison_pvalues.csv', f'{out}/tmp/reward_comparison_pvalues.csv'),
    ('results/sectionP_reward_comparison/reward_comparison_stochastic_pvalues.csv',
     f'{out}/tmp/reward_comparison_stochastic_pvalues.csv'),
]
bad = 0
for a, b in pairs:
    same = os.path.exists(b) and filecmp.cmp(a, b, shallow=False)
    bad += not same
    print(f"  {'identical' if same else 'DIFFERS  '}  {a}")
imgs = sorted(os.listdir('figures/main')) + sorted(os.listdir('figures/supplementary'))
print(f"  {len(imgs)} published figure files; regenerated copies in {out}/ (compare visually: PDFs embed timestamps)")
sys.exit(1 if bad else 0)
PY
echo; echo "done -> $OUT/"
