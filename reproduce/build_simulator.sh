#!/usr/bin/env bash
# Rebuild the simulator (auxiliary models) from the race data in data/. Hours; not needed to
# reproduce the figures (the shipped simulator is in the Zenodo archive).
#
# The auxiliary trainers write the BASE tree to trained_models/simulation, which is also where the
# shipped (final) simulator lives. This script therefore refuses to run while
# trained_models/simulation exists: move it first, e.g.
#     mv trained_models/simulation trained_models/simulation_shipped
# and point back to it afterwards (F1_SIMULATION_DIR) or replace it with the rebuilt one.
#
# Order (the matrix completion reads the coefficients exported by the first regression pass):
#   0. (optional) scripts/build_dataframes.py   FastF1 -> data/session_20{21..24}_V2.csv  (needs fastf1)
#   1. scripts/train_regressions.py             clear-track lap-time regression (Eq. 1), 1st pass
#   2. scripts/run_matrix_completion.py         compound coefficients (Alg. B.1)
#   3. scripts/train_regressions.py             2nd pass with the completed coefficients
#   4. scripts/train_yf_models.py               yellow-flag lap-time regression
#   5. scripts/train_overtake_models.py         overtake models
#   6. scripts/train_rival_models.py            rival pit-stop / compound logits
#   7. src/models/logit_overtakes_pit.py        pit-aware overtake logit  -> trained_models/simulation_pitlogit_20260807 (overlay)
#   8. scripts/build_simulation_yf2024_20260831.py  2024 yellow-flag coefficients (data/yf_regression_2024.xlsx)
#                                               -> trained_models/simulation_yf2024_20260831 (overlay)
#   9. resolve the overlay into a self-contained tree: trained_models/simulation_rebuilt
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduce/env_paper.sh
if [ -e trained_models/simulation ]; then
  echo "trained_models/simulation exists (shipped simulator). Move it first:"
  echo "    mv trained_models/simulation trained_models/simulation_shipped"; exit 1
fi
if [ "${WITH_FASTF1:-0}" = 1 ]; then $PYTHON scripts/build_dataframes.py; fi
# the auxiliary models are fitted on the unmodified environment defaults
( unset F1_DSQ_PENALTY F1_LASTLAP_PITOUT F1_SC_BUNCHING F1_SC_BUNCH_DELTA F1_FOLLOWING_GAP_SCALE
  $PYTHON scripts/train_regressions.py
  $PYTHON scripts/run_matrix_completion.py
  $PYTHON scripts/train_regressions.py
  $PYTHON scripts/train_yf_models.py
  $PYTHON scripts/train_overtake_models.py
  $PYTHON scripts/train_rival_models.py
  $PYTHON src/models/logit_overtakes_pit.py
  $PYTHON scripts/build_simulation_yf2024_20260831.py )
cp -RL trained_models/simulation_yf2024_20260831 trained_models/simulation_rebuilt
echo "rebuilt simulator: trained_models/simulation_rebuilt  (use F1_SIMULATION_DIR=trained_models/simulation_rebuilt)"
