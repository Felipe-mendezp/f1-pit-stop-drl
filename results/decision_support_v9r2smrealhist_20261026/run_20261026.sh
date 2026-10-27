#!/bin/zsh
# 2026-10-26 — E4 + E3 de la propuesta de post-entrenamiento (docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md),
# agentes trained_models/retrained_v9r2_startmix_20261025/selected (regla 5/7/8 + max-dsq 0 entre 3 replicas; E1 en la
# largada real, ver results/startmix_proposal_20261025/README.md). F1_START_MIX apagado al evaluar.
#   E4: barrido B (historia real), mismo protocolo que ../decision_support_v9r2realhist_20261024/run_20261024.sh (M = 500,
#       PPO y RPPO, (i) y (ii), 8 GP) -> esta carpeta; luego el control de mascara (mask_check_20261026.py).
#   E3: Figs. 6-7 (escenarios (i) y (ii), 500 carreras, con QMIP) via reproduce/pipeline_v9r2_20261023.sh fig6/fig7
#       -> results/startmix_proposal_20261025/e3_world2024_20261026/
#   zsh results/decision_support_v9r2smrealhist_20261026/run_20261026.sh
cd "$(dirname $0)/../.."
H=results/decision_support_v9r2smrealhist_20261026
source reproduce/env_paper_v3_20261007.sh
export MPLBACKEND=Agg F1_WORLD2024_TYRES=v7
export F1_RIVAL_START_ALPHA=$(~/.venvs/f1-drl-v2/bin/python -c "import json; print(json.load(open('data/rival_start_alpha_2021_2023_20261018.json'))['alpha'])")
export F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
export F1_TYRE_SETS=$PWD/data/tyre_sets_2024_reg_20261021.csv F1_TYRE_REUSE=1 F1_NO_SET_PENALTY=-1000
unset F1_RIVAL_DNF F1_START_MIX
PY=~/.venvs/f1-drl-v2/bin/python
echo "$(date +%T) E4 inicio"
F1_DS_FORCE=hist $PY scripts/run_ds_v6_sweep_20261015.py --version v9r2smhist --M 500 --workers 8 --figures
rc=$?; echo "$(date +%T) E4 fin rc=$rc"
$PY $H/mask_check_20261026.py > $H/mask_check_20261026.log 2>&1
rc=$?; echo "$(date +%T) mascara rc=$rc"
echo "$(date +%T) E3 inicio"
OUT=results/startmix_proposal_20261025/e3_world2024_20261026 AGENTS=trained_models/retrained_v9r2_startmix_20261025 \
  zsh reproduce/pipeline_v9r2_20261023.sh fig6
rc=$?; echo "$(date +%T) E3 fig6 rc=$rc"
OUT=results/startmix_proposal_20261025/e3_world2024_20261026 AGENTS=trained_models/retrained_v9r2_startmix_20261025 \
  zsh reproduce/pipeline_v9r2_20261023.sh fig7
rc=$?; echo "$(date +%T) E3 fig7 rc=$rc"
echo "$(date +%T) TODO"
