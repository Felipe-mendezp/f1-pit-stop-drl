"""2026-10-30 — Igual que compare_20261030.py pero solo pares sin parada de ninguno de los dos autos (adelantamientos en
pista), con la distribucion de gaps por tramo fino y la descomposicion pares x tasas.
    python results/overtake_calib_20261030/compare_nopit_20261030.py <tag>
"""
import os, sys, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else 'r28_k'
pd.set_option('display.width', 250)
s = pd.read_pickle(os.path.join(HERE, 'real_pairs_ot_gap_20261029.pkl'))
s['pit'] = s.own_pit.astype(bool) | (s.rival_PitIn == 1) | (s.rival_PitOut == 1)
s = s.drop(columns=['GP']).rename(columns={'g': 'GP', 'interval_front_real': 'gap', 'ot_pair': 'ot'}); s['race'] = s.Year
R = pd.concat([pd.read_pickle(f) for f in glob.glob(os.path.join(HERE, 'sim', f'{TAG}__*.pkl'))])
R['delta_total'] = R.gap + R.dmean; R['pit'] = (R.pit_x == 1) | (R.pit_y == 1) | (R.prev_x == 1) | (R.prev_y == 1); R['race'] = R.seed  # parada: vuelta de la decision y la de salida
R = R[(R.lap >= 5) & (R.delta_total >= -30) & (R.delta_total <= 5)]
s, R = s[~s.pit].copy(), R[~R.pit].copy()
B = [-np.inf, 0.5, 1, 1.5, 2, 3, 5, np.inf]
for d in (s, R):
    d['b'] = pd.cut(d.gap, B)
def per_race(d, col=None):
    nr = d.groupby('GP').race.nunique()
    return (d.groupby(['GP', 'b'], observed=False).size() if col is None else d.groupby(['GP', 'b'], observed=False)[col].sum()).unstack().div(nr, axis=0)
print(f'== {TAG}: pares sin parada (adelantamientos en pista), por carrera')
for lab, d in [('real', s), ('sim', R)]:
    print(f'\n-- {lab}: pares por carrera, por tramo de gap al inicio de vuelta')
    print(per_race(d).round(1).to_string())
for lab, d in [('real', s), ('sim', R)]:
    print(f'\n-- {lab}: tasa de adelantamiento por tramo')
    print(d.groupby(['GP', 'b'], observed=False).ot.mean().unstack().round(3).to_string())
rows = []
for g in sorted(s.GP.unique()):
    a, b = s[s.GP == g], R[R.GP == g]
    ca = a.groupby('b', observed=False).size() / a.race.nunique(); cb = b.groupby('b', observed=False).size() / b.race.nunique()
    ra = a.groupby('b', observed=False).ot.mean().fillna(0); rb = b.groupby('b', observed=False).ot.mean().fillna(0)
    rows.append(dict(GP=g, real=(ca * ra).sum(), sim=(cb * rb).sum(), pares_real_tasas_sim=(ca * rb).sum(),
                     pares_sim_tasas_real=(cb * ra).sum(), pares_real=ca.sum(), pares_sim=cb.sum(),
                     le1_real=ca.iloc[:2].sum() / ca.sum(), le1_sim=cb.iloc[:2].sum() / cb.sum()))
X = pd.DataFrame(rows).set_index('GP'); X.loc['Total'] = X.sum(); X.loc['Total', ['le1_real', 'le1_sim']] = np.nan
print('\nDescomposicion (por carrera, sin paradas)'); print(X.round(2).to_string())
