#!/bin/zsh
# 2026-10-28 — Cadena v10 con compuestos corregidos (data_fix_20261028; factor 2021 = 0.74 igual que v10, re-estimado en
# factor2021_v10.log: 0.7418). Orden: neumaticos -> Ec. 1 -> SC/VSC -> rivales -> adelantamiento. Salidas en esta carpeta.
#   zsh results/sim_v10_20261028/run_chain_20261028.sh
cd "$(dirname $0)/../.."
export MPLBACKEND=Agg F1_TYRELIFE_2021=v10
PY=~/.venvs/f1-drl-v2/bin/python
D=results/sim_v10_20261028
for s in tyre laptime yf rivals overtake; do
  echo "== $s $(date +%T)"
  $PY $D/${s}_v10_20261027.py > $D/${s}_v10.log 2>&1 || { echo "FALLA $s"; tail -20 $D/${s}_v10.log; exit 1; }
  grep -v Warn $D/${s}_v10.log | tail -4
done
echo FIN
