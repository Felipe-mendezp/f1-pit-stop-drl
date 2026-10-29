#!/bin/bash
source "results/miami_lap6_diag_20261029/env_seleccion_20261029.sh"
T=results/miami_lap6_diag_20261029; mkdir -p $T/sw
for k in 0 1 2 3 4 5; do
  $PY $WRAP "$T/fixed_sweep_20261029.py" --n 200 --part $k/6 --out "$T/sw/part$k.csv" > "$T/sw/part$k.log" 2>&1 &
done
$PY $WRAP "$T/fixed_sweep_20261029.py" --n 200 --laps 6 --part 0/99 --agents PPO:1,RecurrentPPO:1,DQN:8,A2C:8 --out "$T/sw/agents.csv" > "$T/sw/agents.log" 2>&1 &
wait
echo done
