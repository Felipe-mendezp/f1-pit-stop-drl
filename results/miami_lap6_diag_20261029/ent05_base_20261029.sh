#!/bin/bash
# 2026-10-29 §9: linea base con el mismo evaluador: PPO/RPPO originales (ent_coef 0.02) de Miami, 10 semillas.
cd "$(dirname "$0")/../.."
T=results/miami_lap6_diag_20261029
bash $T/ent05_evaluar_20261029.sh "trained_models/retrained_v10eGQf_20261029/seed{seed:02d}" PPO base_PPO
bash $T/ent05_evaluar_20261029.sh "trained_models/retrained_v10eGQf_20261029/seed{seed:02d}" RecurrentPPO base_RPPO
