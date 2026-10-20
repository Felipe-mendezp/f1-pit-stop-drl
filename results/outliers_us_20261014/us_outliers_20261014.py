"""2026-10-14 — Apendice: tabla de la Ec. 1 antes/despues de outliers y figura de vueltas outlier, para el GP de
Estados Unidos (mismo filtro y mismas etapas (1) y (2) que results/fig3_20261014/plot_fig3_20261014.py).
  antes : todas las vueltas con bandera verde, ediciones 2021-23 sin las mojadas (filtro A), sin el corte de +15 s
  despues: all_laps() (corte de +15 s) + filtro A + IQR iterativo (Algoritmo 1)
Correr desde la raiz del repo. Salida en esta carpeta: table_us.{csv,tex}, laptimes_outliers_us.{pdf,png}
"""
import os, sys, io, contextlib, importlib.util
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import models.regressions as R
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', 'results/tyre_coef_study_20260930/pooled_20261004.py')
with contextlib.redirect_stdout(io.StringIO()):
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GP = 'United States Grand Prix'
raw = pd.concat(R.load_data()[:3]); wet = P.wet_editions(raw)
with contextlib.redirect_stdout(io.StringIO()):
    lw = R.all_laps(outliers=True)
lw = lw[[(y, g) not in wet for y, g in zip(lw.Year, lw.GP)]]
d1 = lw[(lw.GP == GP) & (lw.TrackStatus == 1.0)]
with contextlib.redirect_stdout(io.StringIO()):
    m1, _ = R.calibrar_regresion_gp(d1)
    m2, d2 = R.filtrar_outliers_iqr(P.hist_laps_A(), GP)
print('editions:', sorted(d1.Year.unique()), ' wet excluded:', sorted(w for w in wet if w[1] == GP))
print(f'N {int(m1.nobs)} -> {int(m2.nobs)}  removed {1 - m2.nobs / m1.nobs:.1%}   R2 {m1.rsquared:.3f} -> {m2.rsquared:.3f}')
star = lambda p: '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else ''
def nice(n):
    n = n.replace("C(Driver, Treatment(reference='VER'))[T.", 'drv:').replace(']', '')
    n = n.split('[T.')[-1] if 'Compound_Detail, Treatment' in n else n
    n = 'Year ' + n.split('[T.')[-1] if 'C(Year' in n else n
    n = n.replace('TyreLife:C(Compound_Detail)[', 'TireLife x ')
    return n
T = pd.concat({'with': pd.DataFrame(dict(coef=m1.params, se=m1.bse, p=m1.pvalues)),
               'without': pd.DataFrame(dict(coef=m2.params, se=m2.bse, p=m2.pvalues))}, axis=1)
T.index = [nice(i) for i in T.index]
for k in ('with', 'without'):
    T[(k, 'sig')] = T[(k, 'p')].map(star)
T.to_csv(os.path.join(HERE, 'table_us.csv'))
pd.set_option('display.width', 200)
print(T[[('with', 'coef'), ('with', 'se'), ('with', 'sig'), ('without', 'coef'), ('without', 'se'), ('without', 'sig')]].round(4).to_string())
print('compound ref (with/without):', d1.Compound_Detail.value_counts().idxmax(), d2.Compound_Detail.value_counts().idxmax())
print('laps per compound before/after:', d1.Compound_Detail.value_counts().to_dict(), d2.Compound_Detail.value_counts().to_dict())
# figura: vueltas con bandera verde del GP (antes), marcando las que no quedan en el ajuste final
key = ['Year', 'Driver', 'LapNumber']
kept = set(map(tuple, d2[key].itertuples(index=False)))
g = d1.copy(); g['out'] = [tuple(r) not in kept for r in g[key].itertuples(index=False)]
print(f"outliers in figure: {g.out.sum()}/{len(g)} = {g.out.mean():.1%};  by year:", g.groupby('Year').out.agg(['sum', 'size']).to_dict('index'))
print('outliers by lap (top):', g[g.out].LapNumber.value_counts().head(8).to_dict())
print('negative outliers:', g[g.out & (g.LapTime < g.groupby(['Year'])['LapTime'].transform('median'))][key + ['LapTime', 'Compound_Detail', 'TyreLife']].to_string())
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1, g.Year.nunique(), figsize=(12, 4.2), sharey=True)
for ax, (y, gy) in zip(np.atleast_1d(axs), g.groupby('Year')):
    for _, dd in gy.groupby('Driver'):
        dd = dd.sort_values('LapNumber'); ax.plot(dd.LapNumber, dd.LapTime, color='black', lw=0.3, zorder=1)
    ok, out = gy[~gy.out], gy[gy.out]
    ax.scatter(ok.LapNumber, ok.LapTime, s=14, color='tab:blue', edgecolors='black', linewidths=0.2, label='Non-outlier', zorder=2)
    ax.scatter(out.LapNumber, out.LapTime, s=18, color='tab:orange', edgecolors='black', linewidths=0.2, label='Outlier', zorder=3)
    ax.set_title(str(int(y))); ax.set_xlabel('Lap Number'); ax.grid(alpha=0.3)
np.atleast_1d(axs)[0].set_ylabel('Lap Times [s]'); np.atleast_1d(axs)[-1].legend(loc='upper right')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'laptimes_outliers_us.{ext}'), dpi=200, bbox_inches='tight')
