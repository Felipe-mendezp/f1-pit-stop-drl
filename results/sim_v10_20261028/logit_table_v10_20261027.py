"""2026-10-27 — Filas de la Tabla tab:logit_pitin (suplementario) con el LogitStop v10 (arbol simulation_v10_20261027).
Imprime 'etiqueta & coef & estrellas & (ee)' para las filas que muestra la tabla, N y pseudo R2.
    ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/logit_table_v10_20261027.py
"""
import os, sys, pickle
sys.path[:0] = ['.', 'src']
import config, compat; compat.setup_all()


def load(p):   # dentro de una funcion: patsy necesita un frame que lo llame al deshacer el pickle
    with open(p, 'rb') as fh:
        return pickle.load(fh)


m = load('trained_models/simulation_v10_20261027/Bahrain Grand Prix/rival_pitstop_logit.pkl')
star = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
rows = [('Intercept', 'Intercept')]
for g in ['Bahrain', 'Belgian', 'Dutch', 'Hungarian', 'Miami', 'Saudi Arabian', 'Singapore', 'United States']:
    rows.append((f'{g} GP', f'GP[T.{g} Grand Prix]'))
rows += [('SC', 'SC'), ('VSC', 'VSC'), ('Used two compounds', 'used_two_compounds'), ('Early laps', 'EarlyLaps'),
         ('Late laps', 'LateLaps')]
rows += [(f'TyreLife $\\cdot$ C{c}', f'TyreLife:C(Compound_Detail)[C{c}]') for c in range(1, 6)]
rows += [('Position', 'Position')]
for lab, k in rows:
    print(f'{lab} & {m.params[k]:.4f} & {star(m.pvalues[k])} & ({m.bse[k]:.3f}) \\\\')
print(f'N {int(m.nobs):,} | pseudo R2 {m.prsquared:.3f}')
