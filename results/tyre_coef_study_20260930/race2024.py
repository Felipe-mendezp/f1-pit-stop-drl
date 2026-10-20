"""Ec.1 estimada SOLO con la carrera 2024 de cada GP (vara de validacion, no calibracion).
Mismas transformaciones que all_laps(), mismo filtro (A) (>10 % WET/INT) e IQR iterativo."""
import sys, os
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
out = sys.argv[1]
_, _, _, d = R.load_data()
share = d.groupby('GP')['Compound'].transform(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
print('GP 2024 excluidos por lluvia:', sorted(d.loc[share > 0.10, 'GP'].unique()))
d = R.filter_laps(d[share <= 0.10]).copy()
d['DRS'] = ((d['Interval_front'] < 1.0) & (d['LapNumber'] > 2)).astype(int)
d['Interval_front'] = np.exp(-d['Interval_front']); d['Interval_behind'] = np.exp(-d['Interval_behind'])
d['LapNumber_2'] = d['LapNumber'] ** 2
d['FirstLap_pos'] = d['Position'] * (d['LapNumber'] == 1.0)
med = d.groupby('GP')['LapTime'].transform('median')
d = d[~((d['LapTime'] >= med + 15) & ~((d['PitIn'] == 1) | (d['PitOut'] == 1) | (d['LapNumber'] == 1)))]
rows = []
for gp in d.GP.unique():
    try:
        m, dd = R.filtrar_outliers_iqr(d, gp)
    except Exception as e:
        print('skip', gp, e); continue
    for n in m.params.index:
        if n.startswith('TyreLife:'):
            c = n.split('[')[-1][:-1]
            rows.append({'GP': gp, 'compound': c, 'slope_2024': m.params[n], 'se_2024': m.bse[n],
                         'n_laps': int((dd.Compound_Detail == c).sum())})
pd.DataFrame(rows).to_csv(os.path.join(out, 'race2024_slopes.csv'), index=False)
print('ok', len(rows))
