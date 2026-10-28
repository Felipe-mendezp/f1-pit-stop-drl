"""2026-10-27 — Reestima el modelo YF v4 (scripts/fit_yf_model_v4_20261007.py) con PitIn/PitOut corregidos por cambio de
juego (no escribe nada en data/). Correr desde la raiz del repo."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'scripts')
import fit_yf_model_v4_20261007 as F, pandas as pd, statsmodels.formula.api as smf
d = F.load_laps().sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
k = ['Year', 'GP', 'Driver']; n = d.groupby(k).shift(-1); p = d.groupby(k).shift(1)
d['stop'] = ((n.TyreLife < d.TyreLife) | (n.Compound_Detail != d.Compound_Detail)) & (n.LapNumber == d.LapNumber + 1)
d['stop_prev'] = ((d.TyreLife < p.TyreLife) | (d.Compound_Detail != p.Compound_Detail)) & (p.LapNumber == d.LapNumber - 1)
G = F.green_components(d)
y = d[d.pure.isin(['SC', 'VSC'])].copy()
y['VSC'] = (y.pure == 'VSC').astype(int); y['SCl'] = ((y.pure == 'SC') & (y.ev_idx == 2)).astype(int)
y['SCf'] = ((y.pure == 'SC') & (y.ev_idx >= 3)).astype(int)
y = y[(y.VSC + y.SCl + y.SCf) == 1]; y = y[~(((y.PitIn + y.PitOut) > 0) & (y.SCf == 1))]
y = y.join(G[['Gin', 'Gout']], on='GP').dropna(subset=['Gin', 'Gout'])
y['ratio'] = y.LapTime / y.Lref; y = y[(y.ratio > 0.9) & (y.ratio < 2.2)]
print('muestra', len(y), 'PitIn', y.PitIn.sum(), 'sin parada', ((y.PitIn == 1) & ~y.stop).sum(),
      '| PitOut', y.PitOut.sum(), 'sin parada', ((y.PitOut == 1) & ~y.stop_prev).sum())
for name, pi, po in [('original', y.PitIn, y.PitOut), ('corregido', (y.PitIn == 1) & y.stop, (y.PitOut == 1) & y.stop_prev)]:
    z = y.assign(PI=pi * y.Gin / y.Lref, PO=po * y.Gout / y.Lref, SC=y.SCl + y.SCf,
                 rl=y.Year.astype(str) + '|' + y.GP + '|' + y.LapNumber.astype(int).astype(str))
    m = smf.mixedlm('ratio ~ 0 + VSC + SCl + SCf + PI + PO', z, groups=z['GP'], re_formula='~0+SC',
                    vc_formula={'rl': '0+C(rl)'}).fit(method='lbfgs')
    print(name, m.fe_params.round(3).to_dict(), m.bse.iloc[:5].round(3).to_dict())
