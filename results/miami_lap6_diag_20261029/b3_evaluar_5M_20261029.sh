#!/bin/bash
# 2026-10-29 B3: 200 carreras de validacion (mundo de la seleccion) para cada best_model de la prueba de entropia.
cd "$(dirname "$0")/../.."
source results/miami_lap6_diag_20261029/env_seleccion_20261029.sh
T=results/miami_lap6_diag_20261029
mkdir -p $T/b3_5M
for ENT in 0.05 0.10; do for S in 1 2; do
  $PY $WRAP $T/fixed_sweep_20261029.py --n 200 --laps 6 --part 0/99 --agents PPO:$S \
     --tree "trained_models/probe_miami_entropy5M_20261029/ent${ENT}_seed{seed:02d}" \
     --out $T/b3_5M/ent${ENT}_seed0$S.csv > $T/b3_5M/ent${ENT}_seed0$S.log 2>&1 &
done; done
wait; echo fin
