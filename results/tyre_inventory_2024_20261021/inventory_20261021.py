"""Reconstruction of each driver's tyre sets before the 2024 race (8 GPs of the paper) from FastF1 laps.

Input: laps_2024_8gp.csv (download_20261021.py: every session of the weekend, laps only).
Per driver and session (in order), each stint gives (compound, TyreLife at its first and last lap, FreshTyre).
FastF1's TyreLife carries the laps of earlier sessions, so:
  - a stint that starts on a fresh tyre opens a new set;
  - a stint on a used tyre is assigned to the set of the same compound whose last TyreLife is the largest one not
    above the stint's starting TyreLife (ties: the most recently used set); if none fits, a set is created as
    'unmatched' (used before the data starts, or a matching failure).
Before the race: sets opened per compound and their laps; validation: every race stint that starts on a used set
(lap 1 or out-lap) should match a pre-race set of the same compound (TyreLife at the start of the race stint
>= last pre-race TyreLife of that set).
    python inventory_20261021.py laps_2024_8gp.csv
"""
import sys
import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1])
d = d[d.Compound.isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
race_name = 'Race'
sets_rows, val_rows = [], []
for (gp, drv), g in d.groupby(['GP', 'Driver']):
    sets = []  # dicts: compound, sid, life, first_session, uses (list of (session, start, end))
    pre = g[g.Session != race_name].sort_values(['SessionOrder', 'LapNumber'])
    for (so, ses, st), s in pre.groupby(['SessionOrder', 'Session', 'Stint'], sort=True):
        s = s.dropna(subset=['TyreLife'])
        if s.empty:
            continue
        comp = s.Compound.iloc[0]; start, end = float(s.TyreLife.iloc[0]), float(s.TyreLife.max())
        fresh = bool(s.FreshTyre.iloc[0]) if s.FreshTyre.notna().iloc[0] else start <= 1
        cand = [x for x in sets if x['compound'] == comp and x['life'] <= start] if not fresh else []
        if cand:
            best = max(cand, key=lambda x: (x['life'], x['last']))
        else:
            best = dict(compound=comp, sid=len(sets) + 1, life=0.0, first=ses,
                        status='new' if fresh else 'unmatched', uses=[], last=0)
            sets.append(best)
        best['life'] = end; best['last'] = so
        best['uses'].append(f"{ses}: {int(start)}-{int(end)}")
    for x in sets:
        sets_rows.append(dict(GP=gp, Driver=drv, Set=x['sid'], Compound=x['compound'], Opened=x['first'],
                              Origin=x['status'], LapsBeforeRace=int(x['life']), Uses=' | '.join(x['uses'])))
    race = g[g.Session == race_name].sort_values('LapNumber')
    for st, s in race.groupby('Stint'):
        s = s.dropna(subset=['TyreLife'])
        if s.empty:
            continue
        comp = s.Compound.iloc[0]; start = float(s.TyreLife.iloc[0])
        fresh = bool(s.FreshTyre.iloc[0]) if s.FreshTyre.notna().iloc[0] else start <= 1
        if fresh:
            continue
        lap = int(s.LapNumber.iloc[0])
        ok = [x for x in sets if x['compound'] == comp and x['life'] <= start]
        gap = min((start - x['life'] for x in ok), default=np.nan)
        val_rows.append(dict(GP=gp, Driver=drv, RaceStint=int(st), StartLap=lap, Compound=comp,
                             TyreLifeAtStart=int(start), Matched=bool(ok), Gap=gap))
S = pd.DataFrame(sets_rows); V = pd.DataFrame(val_rows)
S.to_csv('inventory_2024_8gp.csv', index=False); V.to_csv('inventory_validation_2024_8gp.csv', index=False)
print(f"{len(S)} juegos reconstruidos; origen: {S.Origin.value_counts().to_dict()}")
print('juegos abiertos antes de la carrera por compuesto (mediana por piloto):')
print(S.groupby(['GP', 'Driver', 'Compound']).size().unstack(fill_value=0).groupby('GP').median().to_string())
if len(V):
    print(f"\nvalidacion: {len(V)} tandas de carrera con juego usado; con juego previo compatible: {V.Matched.mean():.0%}; "
          f"diferencia de vida (vueltas) mediana {V.Gap.median():.0f}, p90 {V.Gap.quantile(.9):.0f}")
    print(V.groupby('GP').Matched.agg(['size', 'mean']).round(2).to_string())
