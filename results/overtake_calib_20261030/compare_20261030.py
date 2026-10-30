"""2026-10-30 — Adelantamientos simulados vs reales con la MISMA muestra que la estimacion del logit:
pares adyacentes, gap al inicio de la vuelta, vuelta siguiente sin SC/VSC, L >= 4 (sim: vuelta >= 5), delta_total =
gap + diferencia de tiempos medios en [-30, 5]. Real: real_pairs_ot_gap_20261029.pkl (copia de ot_gap.pkl, 2021-23).
    python results/overtake_calib_20261030/compare_20261030.py <tag>
"""
import os, sys, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else 'r28_k'
pd.set_option('display.width', 250)
s = pd.read_pickle(os.path.join(HERE, 'real_pairs_ot_gap_20261029.pkl'))
s['pit'] = s.own_pit.astype(bool) | (s.rival_PitIn == 1) | (s.rival_PitOut == 1)
s = s.drop(columns=['GP', 'gap'], errors='ignore').rename(columns={'g': 'GP', 'interval_front_real': 'gap', 'ot_pair': 'ot'}); s['race'] = s.Year
R = pd.concat([pd.read_pickle(f) for f in glob.glob(os.path.join(HERE, 'sim', f'{TAG}__*.pkl'))]); R['GP'] = R.GP.str.strip()
R['delta_total'] = R.gap + R.dmean; R['pit'] = (R.pit_x == 1) | (R.pit_y == 1) | (R.prev_x == 1) | (R.prev_y == 1); R['race'] = R.seed  # parada: vuelta de la decision y la de salida
R_all = R[R.lap >= 5]
R = R_all[(R_all.delta_total >= -30) & (R_all.delta_total <= 5)]
band = lambda g: pd.cut(g, [-np.inf, 1, 2, 5, np.inf], labels=['<=1', '1-2', '2-5', '>5'])
def summ(x, lab):
    nr = x.groupby('GP').race.nunique()
    o = pd.DataFrame({'filas': x.groupby('GP').size() / nr, 'ot': x.groupby('GP').ot.sum() / nr,
                      'ot_sinpit': x[~x.pit].groupby('GP').ot.sum() / nr, 'ot_pit': x[x.pit].groupby('GP').ot.sum() / nr,
                      'le1': x.groupby('GP').gap.apply(lambda g: (g <= 1).mean()),
                      'tasa_le1': x[x.gap <= 1].groupby('GP').ot.mean(),
                      'tasa_1_2': x[(x.gap > 1) & (x.gap <= 2)].groupby('GP').ot.mean(),
                      'tasa_gt2': x[x.gap > 2].groupby('GP').ot.mean()})
    o.columns = pd.MultiIndex.from_product([[lab], o.columns]); return o
T = pd.concat([summ(s, 'real'), summ(R, 'sim')], axis=1)
print(f'== {TAG}: por carrera, misma muestra que el logit (delta_total en [-30, 5])')
print(T.round(3).to_string())
tot = lambda x: (x.groupby('GP').ot.sum() / x.groupby('GP').race.nunique()).sum()
print(f'\nTotal 8 GP: real {tot(s):.1f} | sim misma muestra {tot(R):.1f} | sim sin filtro delta_total (conteo anterior) {tot(R_all):.1f}')
# descomposicion: cuantos adelantamientos daria el sim con la distribucion de gaps real (tasas sim por tramo) y viceversa
for lab, d in [('real', s), ('sim', R)]:
    d['b'] = band(d.gap)
rows = []
for g in sorted(s.GP.unique()):
    a, b = s[s.GP == g], R[R.GP == g]
    na, nb = a.race.nunique(), b.race.nunique()
    ca = a.groupby('b', observed=False).size() / na; cb = b.groupby('b', observed=False).size() / nb
    ra = a.groupby('b', observed=False).ot.mean().fillna(0); rb = b.groupby('b', observed=False).ot.mean().fillna(0)
    rows.append(dict(GP=g, real=(ca * ra).sum(), sim=(cb * rb).sum(), pares_real_tasas_sim=(ca * rb).sum(),
                     pares_sim_tasas_real=(cb * ra).sum()))
print('\nDescomposicion (por carrera): pares por tramo de gap x tasa por tramo')
print(pd.DataFrame(rows).set_index('GP').round(1).to_string())
