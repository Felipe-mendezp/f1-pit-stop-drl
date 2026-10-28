"""2026-10-27 — Modelo YF v4 eliminando las vueltas de paso por el pit lane sin parar (PitIn=1 y la vuelta siguiente con
el mismo compuesto y TyreLife+1; PitOut idem con la anterior). Opcion aprobada por el autor para v10. No escribe en data/.
    ~/.venvs/f1-drl-v2/bin/python results/pitin_flag_check_20261027/05_yf_v4_drop_pass_20261027.py"""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'scripts')
import fit_yf_model_v4_20261007 as F, pandas as pd, statsmodels.formula.api as smf
d = F.load_laps().sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
k = ['Year', 'GP', 'Driver']; n = d.groupby(k).shift(-1); p = d.groupby(k).shift(1)
d['in_pass'] = (d.PitIn == 1) & (n.LapNumber == d.LapNumber + 1) & (n.Compound_Detail == d.Compound_Detail) & (n.TyreLife == d.TyreLife + 1)
d['out_pass'] = (d.PitOut == 1) & (p.LapNumber == d.LapNumber - 1) & (p.Compound_Detail == d.Compound_Detail) & (d.TyreLife == p.TyreLife + 1)
G = F.green_components(d[~(d.in_pass | d.out_pass)])
def sample(d):
    y = d[d.pure.isin(['SC', 'VSC'])].copy()
    y['VSC'] = (y.pure == 'VSC').astype(int); y['SCl'] = ((y.pure == 'SC') & (y.ev_idx == 2)).astype(int)
    y['SCf'] = ((y.pure == 'SC') & (y.ev_idx >= 3)).astype(int)
    y = y[(y.VSC + y.SCl + y.SCf) == 1]; y = y[~(((y.PitIn + y.PitOut) > 0) & (y.SCf == 1))]
    y = y.join(G[['Gin', 'Gout']], on='GP').dropna(subset=['Gin', 'Gout'])
    y['ratio'] = y.LapTime / y.Lref; return y[(y.ratio > 0.9) & (y.ratio < 2.2)]
for name, D in [('original', d), ('sin pasos por el pit lane', d[~(d.in_pass | d.out_pass)])]:
    y = sample(D)
    z = y.assign(PI=y.PitIn * y.Gin / y.Lref, PO=y.PitOut * y.Gout / y.Lref, SC=y.SCl + y.SCf,
                 rl=y.Year.astype(str) + '|' + y.GP + '|' + y.LapNumber.astype(int).astype(str))
    m = smf.mixedlm('ratio ~ 0 + VSC + SCl + SCf + PI + PO', z, groups=z['GP'], re_formula='~0+SC',
                    vc_formula={'rl': '0+C(rl)'}).fit(method='lbfgs')
    print(f"{name}: n={len(y)} PitIn={int(y.PitIn.sum())} PitOut={int(y.PitOut.sum())}", m.fe_params.round(3).to_dict(), m.bse.iloc[:5].round(3).to_dict())
