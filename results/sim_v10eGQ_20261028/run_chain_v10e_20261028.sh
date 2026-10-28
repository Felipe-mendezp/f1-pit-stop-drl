#!/bin/zsh
# 2026-10-28 — Cadena v10eGQ = v10d con la edad efectiva TL_GQ (age_fix_v10e_20261028). Antes: cadena v10d = v10 con compuestos corregidos (data_fix_v10d_20261028: v10c + Belgica y Monaco 2022; factor 2021 = 0.74 igual que v10, re-estimado en
# factor2021_v10.log: 0.7418). Orden: neumaticos -> Ec. 1 -> SC/VSC -> rivales -> adelantamiento. Salidas en esta carpeta.
#   zsh results/sim_v10eGQ_20261028/run_chain_v10e_20261028.sh
cd "$(dirname $0)/../.."
export MPLBACKEND=Agg F1_TYRELIFE_2021=v10eGQ
PY=~/.venvs/f1-drl-v2/bin/python
D=results/sim_v10eGQ_20261028
for s in tyre laptime yf rivals; do
  echo "== $s $(date +%T)"
  $PY $D/${s}_v10_20261027.py > $D/${s}_v10.log 2>&1 || { echo "FALLA $s"; tail -20 $D/${s}_v10.log; exit 1; }
  grep -v Warn $D/${s}_v10.log | tail -4
done
# 2026-10-29: adelantamiento con etiqueta por par y vuelta L+1 sin SC/VSC/roja (antes overtake_v10_20261027.py)
echo "== overtake $(date +%T)"
$PY $D/overtake_v10f_20261029.py > $D/overtake_v10f.log 2>&1 || { echo "FALLA overtake"; tail -20 $D/overtake_v10f.log; exit 1; }
grep -v Warn $D/overtake_v10f.log | tail -4
echo FIN
