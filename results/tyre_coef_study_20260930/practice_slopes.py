"""(B) Pendientes de degradacion 2024 desde FP1-FP3 + Sprint.
LapTime_adj = LapTime - g_GP * LapNumber, con g_GP = pendiente media de LapNumber
de las carreras 2021-2023 del circuito (combustible + evolucion de pista).
Modelo: LapTime_adj ~ C(tanda) + TyreLife:C(Compound_Detail), IQR iterativo k=3.
Uso: python practice_slopes.py <practice_csv> <race_params.csv> <outdir>"""
import sys, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

src, race_params, out = sys.argv[1:4]
WET_MAX, PUSH_MAX, MIN_STINT, K, MAX_IT = 0.10, 1.04, 5, 3, 10

d = pd.read_csv(src, low_memory=False)
log = []
def note(msg, df):
    log.append((msg, len(df))); print(f'{msg:55s} {len(df):7d}')
note('laps totales', d)

# sesiones con lluvia o neumaticos de lluvia (> 10 %) fuera; luego vueltas con lluvia
g = d.groupby(['GP', 'Session'])
wet = g['Compound'].transform(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
rain = g['Rainfall'].transform(lambda x: x.astype(str).eq('True').mean())
excl = d.loc[(wet > WET_MAX) | (rain > WET_MAX), ['GP', 'Session']].drop_duplicates()
print('sesiones excluidas por lluvia:\n' + excl.to_string(index=False))
d = d[(wet <= WET_MAX) & (rain <= WET_MAX) & ~d['Rainfall'].astype(str).eq('True')]
d = d[d['Compound'].isin(['SOFT', 'MEDIUM', 'HARD']) & d['Compound_Detail'].notna()]
note('secas, compuestos S/M/H', d)
d = d[d['LapTime'].notna() & (d['PitIn'] == 0) & (d['PitOut'] == 0)]
d = d[d['TrackStatus'].astype(str).str.strip() == '1']
d = d[d['IsAccurate'].astype(str).eq('True') & ~d['Deleted'].astype(str).eq('True')]
note('vueltas validas (sin pit, pista verde, precisas)', d)
d['tanda'] = d['GP'] + '|' + d['Session'] + '|' + d['Driver'] + '|' + d['Stint'].astype(str)
best = d.groupby('tanda')['LapTime'].transform('min')
d = d[d['LapTime'] <= PUSH_MAX * best]
note(f'vueltas lanzadas (<= {PUSH_MAX:.0%} de la mejor de la tanda)', d)
d = d[d.groupby('tanda')['LapTime'].transform('size') >= MIN_STINT]
note(f'en tandas largas (>= {MIN_STINT} vueltas)', d)

# identificacion: TyreLife - LapNumber constante dentro de la tanda?
dev = (d['TyreLife'] - d['LapNumber']).groupby(d['tanda']).transform(lambda x: x.max() - x.min())
print(f'tandas con TyreLife-LapNumber NO constante: {d.loc[dev > 0, "tanda"].nunique()} de {d["tanda"].nunique()}')

rp = pd.read_csv(race_params).set_index('GP')
rp['g'] = rp['LapNumber'] + rp['LapNumber_2'] * (rp['maxlap'] + 1)

rows = []
for gp, dg in d.groupby('GP'):
    if gp not in rp.index:
        continue
    dg = dg.copy()
    dg['y'] = dg['LapTime'] - rp.at[gp, 'g'] * dg['LapNumber']
    it = 0
    while True:
        m = smf.ols('y ~ C(tanda) + TyreLife:C(Compound_Detail)', data=dg).fit()
        q1, q3 = np.percentile(m.resid, [25, 75]); r = K * (q3 - q1)
        keep = (m.resid >= q1 - r) & (m.resid <= q3 + r)
        if keep.all() or it >= MAX_IT:
            break
        dg = dg.loc[keep[keep].index]; it += 1
    for c in sorted(dg['Compound_Detail'].unique()):
        n = f'TyreLife:C(Compound_Detail)[{c}]'
        if n not in m.params:
            continue
        sub = dg[dg['Compound_Detail'] == c]
        rows.append({'GP': gp, 'compound': c, 'slope_prac': m.params[n], 'se_prac': m.bse[n],
                     'p_prac': m.pvalues[n], 'raw_slope': m.params[n] + rp.at[gp, 'g'],
                     'n_laps': len(sub), 'n_tandas': sub['tanda'].nunique(),
                     'sessions': ','.join(sorted(sub['Session'].str.replace('Practice ', 'FP').unique())),
                     'fuel_g': rp.at[gp, 'g'], 'iqr_iter': it})
res = pd.DataFrame(rows)
os.makedirs(out, exist_ok=True)
res.to_csv(os.path.join(out, 'practice_slopes.csv'), index=False)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
