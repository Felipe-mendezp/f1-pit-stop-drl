"""2026-10-14 — Fig. 3 con la calibracion v5h/v6: curvas de neumatico de un GP (por defecto United States, 56 vueltas;
--gp/--laps para otro) en tres etapas, mismo estilo que
scripts/plot_fig3_facet_compounds.py (FacetGrid, Hard negro, Medium dorado, Soft rojo, y = offset + vueltas*pendiente,
relativo al duro nuevo).
  (1) Ec. 1 por circuito con todas las vueltas (sin quitar outliers), datos 2021-23 con filtro A (sin 2021, mojado)
  (2) Ec. 1 por circuito tras el IQR iterativo (Algoritmo 1), mismo filtro (pooled_20261004.fit_gp)
  (3) modelo conjunto opcion3h (lo que usa el simulador: results/tyre_coef_study_20260930/pooled_hier_20261011)
Correr desde la raiz del repo. Salida: results/fig3_20261014/facet_compounds.{pdf,png} y coefs.csv
"""
import os, sys, io, contextlib, importlib.util
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import models.regressions as R
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', 'results/tyre_coef_study_20260930/pooled_20261004.py')
with contextlib.redirect_stdout(io.StringIO()):
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--gp', default='United States Grand Prix'); _ap.add_argument('--laps', type=int, default=56)
_a = _ap.parse_args()
GP, NLAPS = _a.gp, _a.laps
COMPS = dict(zip(P.MAP[GP], ['Hard', 'Medium', 'Soft']))
COLORS = {'Hard': 'black', 'Medium': '#daa520', 'Soft': 'red'}
STAGES = ['With outliers', 'Without outliers', 'Joint model']

raw = pd.concat(R.load_data()[:3]); wet = P.wet_editions(raw)
with contextlib.redirect_stdout(io.StringIO()):
    lw = R.all_laps(outliers=True)
lw = lw[[(y, g) not in wet for y, g in zip(lw.Year, lw.GP)]]
d = lw[(lw.GP == GP) & (lw.TrackStatus == 1.0)]
with contextlib.redirect_stdout(io.StringIO()):
    m, _ = R.calibrar_regresion_gp(d)
pr = m.params
ref = [c for c in COMPS if not any(f'[T.{c}]' in p for p in pr.index if 'Compound_Detail, Treatment' in p)][0]
c1 = {c: (0.0 if c == ref else pr[f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"]) for c in COMPS}
s1 = {c: pr[f"TyreLife:C(Compound_Detail)[{c}]"] for c in COMPS}
H = P.fit_gp(P.hist_laps_A(), GP); b = dict(zip(H['kinds'], H['b']))
c2 = {c: (0.0 if c == H['ref'] else b[('off', c)]) for c in COMPS}; s2 = {c: b[('sl', c)] for c in COMPS}
J = pd.read_csv('results/tyre_coef_study_20260930/pooled_hier_20261011/coefs_opcion3h.csv')
J = J[J.GP == GP].set_index('compound')
c3 = J.offset_vs_ref.to_dict(); s3 = J.slope.to_dict()
rows, coefs = [], []
for st, (c, s) in zip(STAGES, [(c1, s1), (c2, s2), (c3, s3)]):
    base = c[list(COMPS)[0]]
    for comp, lab in COMPS.items():
        coefs.append(dict(stage=st, compound=comp, offset_vs_hard=c[comp] - base, slope=s[comp]))
        rows += [dict(Lap=l, LapTime=c[comp] - base + l * s[comp], Compound=lab, Scenario=st) for l in range(1, NLAPS + 1)]
pd.DataFrame(coefs).to_csv(os.path.join(HERE, f"coefs_{GP.split(' Grand')[0].replace(' ', '_')}.csv"), index=False)
print(pd.DataFrame(coefs).round(4).to_string(index=False))
df = pd.DataFrame(rows)
g = sns.FacetGrid(df, col='Scenario', col_order=STAGES, height=3, aspect=1, sharey=True)
g.map_dataframe(sns.lineplot, x='Lap', y='LapTime', hue='Compound', palette=COLORS, linewidth=2)
g.set_axis_labels('Tire Usage [laps]', 'Additional Lap Time [s]')
g.set_titles('{col_name}')
for ax in g.axes.flatten():
    ax.set_xlim(0, NLAPS); ax.grid(True, alpha=0.3)
g.add_legend(title='Compound')
sns.move_legend(g, loc='center left', bbox_to_anchor=(0.9, 0.5), frameon=True, title='Compound')
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(HERE, f"facet_compounds_{GP.split(' Grand')[0].replace(' ', '_')}.{ext}"), bbox_inches='tight', dpi=200)
print('->', HERE)
