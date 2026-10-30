#!/bin/zsh
# 2026-10-30 — validacion con la estrategia real (fig7, 8 GP, 500 carreras) para las variantes dadas, 4 procesos
cd "$(dirname $0)"
for v in "$@"; do for g in Bahrain_ALO Saudi_Arabian_NOR Miami_OCO Hungarian_TSU Belgian_GAS Dutch_STR Singapore_HUL United_States_MAG; do
  echo "$v $g"; done; done | xargs -P 4 -n 2 sh -c './run_variant_20261030.sh $0 ${SCEN:-fig7} $1 500'
