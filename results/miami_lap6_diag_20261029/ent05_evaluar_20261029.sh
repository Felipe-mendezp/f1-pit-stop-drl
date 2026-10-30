#!/bin/bash
# 2026-10-29 §9: 200 carreras de validacion de Miami (mundo de la seleccion, mismas que §4) para las 10 semillas de un arbol.
#   bash results/miami_lap6_diag_20261029/ent05_evaluar_20261029.sh <arbol con {seed:02d}> <PPO|RecurrentPPO> <etiqueta>
cd "$(dirname "$0")/../.."
source results/miami_lap6_diag_20261029/env_seleccion_20261029.sh
T=results/miami_lap6_diag_20261029; TREE="$1"; AL="$2"; TAG="$3"
mkdir -p $T/ent05
for S in 1 2 3 4 5 6 7 8 9 10; do
  O=$T/ent05/${TAG}_seed$(printf %02d $S)
  [ -s $O.csv ] && continue
  $PY $WRAP $T/fixed_sweep_20261029.py --n 200 --laps 6 --part 0/99 --agents $AL:$S --tree "$TREE" --out $O.csv > $O.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 2; done
done
wait; echo fin $TAG
