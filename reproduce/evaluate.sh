#!/usr/bin/env bash
# Full re-evaluation of the shipped agents (no training). Runs on COPIES of the model trees so
# the shipped caches in trained_models/ are never overwritten, then compares the new result CSVs
# with the published ones in results/.
#
#   bash reproduce/evaluate.sh fig7     # real-2024 replay, 9 GPs x 6 methods x 500 races   (~45 min)
#   bash reproduce/evaluate.sh fig6     # stochastic races, 9 GPs x 6 methods x 500 races   (~2 h)
#   bash reproduce/evaluate.sh ds       # decision-support rollouts, Singapore 2024 laps 1-61 (Figs. 8-10, hours)
#   bash reproduce/evaluate.sh reward   # reward-function comparison, Section P            (~1.5 h)
#   bash reproduce/evaluate.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env_paper.sh
WHAT="${1:-all}"
EVAL=trained_models/rl_agents_reeval
EVALW=trained_models/rl_agents_reward_reeval
copy_models() {  # model files only (no cached results)
  mkdir -p "$2"
  rsync -a --exclude '*.csv' --exclude '*.tex' --exclude '*.pdf' --exclude '*.png' "$1/" "$2/"
}
compare() {  # compare two result CSVs row by row
  $PYTHON - "$1" "$2" <<'PY'
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
same = a.shape == b.shape and a.reset_index(drop=True).equals(b.reset_index(drop=True))
print(f"  {'IDENTICAL' if same else 'DIFFERS'}: {sys.argv[2]}  vs  {sys.argv[1]}")
if not same and 'final_position' in a and 'final_position' in b:
    print('  mean final position (published / re-evaluated):')
    print(pd.concat([a.groupby('algo').final_position.mean(), b.groupby('algo').final_position.mean()], axis=1).round(3).to_string())
PY
}

if [[ "$WHAT" == fig7 || "$WHAT" == all ]]; then
  echo "=== Fig. 7 (real 2024 replay)"
  copy_models trained_models/rl_agents "$EVAL"
  F1_RL_AGENTS_DIR="$EVAL" $PYTHON scripts/evaluate_all_gps_with_qmip.py --force
  compare results/fig07_real_2024/all_gps_2024_qmip_results.csv "$EVAL/all_gps_2024_qmip_results.csv"
fi
if [[ "$WHAT" == fig6 || "$WHAT" == all ]]; then
  echo "=== Fig. 6 (500 stochastic races)"
  copy_models trained_models/rl_agents "$EVAL"
  F1_RL_AGENTS_DIR="$EVAL" $PYTHON scripts/evaluate_qmip_vs_drl.py --n-sims 500 --force
  compare results/fig06_stochastic/qmip_vs_drl_stochastic_results.csv "$EVAL/qmip_vs_drl_stochastic_results.csv"
fi
if [[ "$WHAT" == ds || "$WHAT" == all ]]; then
  echo "=== Figs. 8-10 (decision-support rollouts, PPO, Singapore 2024, M=500 per action)"
  RD=results/decision_support_reeval; mkdir -p "$RD"
  cp results/decision_support/initial_compounds.json "$RD/"
  $PYTHON scripts/decision_support_rollouts.py --gp 'Singapore Grand Prix' --driver Driver_HUL \
      --model-dir Singapore_HUL --algo PPO --M 500 --fast-env --legal-mask --real-2024 --sim-future \
      --following-gap-scale 1.0 --out-dir "$RD" --laps $(seq 1 61)
  compare results/decision_support/singapore_hul_real_lap18.csv "$RD/singapore_hul_real_lap18.csv"
fi
if [[ "$WHAT" == reward || "$WHAT" == all ]]; then
  echo "=== Section P (reward comparison; each reward keeps its own disqualification penalty)"
  copy_models trained_models/rl_agents_reward "$EVALW"
  ( unset F1_DSQ_PENALTY
    export F1_RL_AGENTS_DIR="$EVALW" F1_RL_REWARD_DIR="$EVALW"
    $PYTHON scripts/evaluate_reward_comparison.py --n-sims 500 --force
    $PYTHON scripts/evaluate_reward_comparison_stochastic.py --n-sims 500 --force )
  compare results/sectionP_reward_comparison/reward_comparison_results.csv "$EVALW/reward_comparison_results.csv"
  compare results/sectionP_reward_comparison/reward_comparison_stochastic_results.csv "$EVALW/reward_comparison_stochastic_results.csv"
fi
