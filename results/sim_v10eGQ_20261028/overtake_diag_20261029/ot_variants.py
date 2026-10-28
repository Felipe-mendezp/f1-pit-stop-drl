"""Variantes de muestra del logit de adelantamiento (sobre la muestra exacta de ot_diag.py)."""
import os, numpy as np, pandas as pd, statsmodels.api as sm
df = pd.read_pickle(os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/ot_df.pkl'))
bad = lambda s: s.str.replace('.0', '', regex=False).str.contains('[4567]')
df['ts_L'] = df.TrackStatus.astype(str)
df['nosc_L'] = ~bad(df.ts_L); df['nosc_L1'] = ~bad(df.ts_L1)
df['green_L'] = df.TrackStatus == 1.0
df['green_L1'] = df.ts_L1.isin(['1.0', '1'])
# etiqueta sin condicion de estado de pista (misma regla que optimize_f1_analysis)
full = pd.read_pickle(os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/ot_full.pkl'))
idx = full.set_index(['Year', 'GP', 'LapNumber', 'Driver'])
npos = idx.next_position.to_dict(); last = idx.LastLap.to_dict()
r_np = np.array([npos.get((y, g, l, r), np.nan) for y, g, l, r in zip(df.Year, df.GP, df.LapNumber, df.rival)])
r_last = np.array([last.get((y, g, l, r), np.nan) for y, g, l, r in zip(df.Year, df.GP, df.LapNumber, df.rival)])
df['ot_any'] = ((df.LapNumber > 1) & (df['position_gain/loss'] > 0) & df.next_position.notna()
                & (r_np >= df.next_position) & (r_last != df.LapNumber)).astype(int)
chk = df[df.green_L]
bad_=chk[chk.ot_any != chk.overtake]; print(len(bad_)); print(bad_[['Year','GP','Driver','LapNumber','Position','next_position','overtake','ot_any']].head())
print('etiqueta reconstruida == original en vueltas L verdes:', bool((chk.ot_any == chk.overtake).all()),
      '| original con L no verde siempre 0:', int(df[~df.green_L].overtake.sum()) == 0)
V = {
    'A actual (todas, etiqueta exige L==1)': (np.ones(len(df), bool), 'overtake'),
    'B solo L verde (==1)': (df.green_L.values, 'overtake'),
    'C solo L+1 sin SC/VSC/roja': (df.nosc_L1.values, 'ot_any'),
    'D L y L+1 sin SC/VSC/roja': ((df.nosc_L & df.nosc_L1).values, 'ot_any'),
}
GPS = ['Bahrain', 'Saudi Arabian', 'Miami', 'Hungarian', 'Belgian', 'Dutch', 'Singapore', 'United States']
D = np.array([-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20])
rows = []
for name, (m, lab) in V.items():
    s = df[m]
    for g in GPS:
        x = s[s.GP == g + ' Grand Prix']
        f = sm.Logit(x[lab].values, sm.add_constant(x.delta_total.values)).fit(disp=False)
        a, b = f.params
        rows.append(dict(var=name[0], GP=g, n=len(x), ot=int(x[lab].sum()), const=a, slope=b, se_slope=f.bse[1],
                         **{f'{d:+d}': 1 / (1 + np.exp(-(a + b * d))) for d in D}))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(R.pivot(index='GP', columns='var', values='slope').loc[GPS].round(3).to_string())
print(R.pivot(index='GP', columns='var', values='n').loc[GPS].to_string())
for v in 'BC':
    print(f'\n== variante {v}: probabilidad (%)')
    print((R[R['var'] == v].set_index('GP')[[f'{d:+d}' for d in D]] * 100).round(1).to_string())
R.to_csv(os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/ot_variants.csv'), index=False)
