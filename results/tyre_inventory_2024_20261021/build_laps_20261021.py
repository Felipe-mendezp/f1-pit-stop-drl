"""Joins the laps used by inventory_20261021.py -> laps_2024_8gp.csv (run from this folder).
FP1-FP3/Sprint: data/practice_laps_2024_20260930.csv; Qualifying: quali_race_2024_8gp.csv (load_quali_20261021.py);
race: data/session_2024_V2.csv (stint = 1 + cumulative PitOut; FreshTyre = TyreLife <= 1)."""
import os
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
       'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
C = ['GP', 'Driver', 'Session', 'Stint', 'LapNumber', 'Compound', 'TyreLife', 'FreshTyre']
p = pd.read_csv(f'{R}/data/practice_laps_2024_20260930.csv', low_memory=False); p = p[p.GP.isin(GPS)]
q = pd.read_csv('quali_race_2024_8gp.csv'); q = q[q.Session == 'Qualifying']
pre = pd.concat([p[C], q[C]])
pre['SessionOrder'] = pre.Session.map({'Practice 1': 1, 'Practice 2': 2, 'Practice 3': 3, 'Sprint': 3, 'Qualifying': 4})
r = pd.read_csv(f'{R}/data/session_2024_V2.csv', low_memory=False)
r = r[r.GP.isin(GPS)].sort_values(['GP', 'Driver', 'LapNumber']).copy()
r['Stint'] = r.groupby(['GP', 'Driver']).PitOut.cumsum() + 1
r['Session'] = 'Race'; r['SessionOrder'] = 9; r['FreshTyre'] = r.TyreLife <= 1
pd.concat([pre, r[C + ['SessionOrder']]]).to_csv('laps_2024_8gp.csv', index=False)
