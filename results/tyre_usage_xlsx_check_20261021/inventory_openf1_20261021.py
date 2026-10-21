"""Pre-race tyre-set inventory for every 2024 GP from the 'OpenF1 Stints' sheet of data/f1_2024_tire_usage.xlsx.

Same assignment rule as ../tyre_inventory_2024_20261021/inventory_20261021.py (FastF1), applied to OpenF1 stints:
  - OpenF1's lap_end of a stint equals lap_start of the next one in the same session (the boundary lap is counted
    twice), so the laps of a non-final stint are next_start - start; the last stint keeps end - start + 1;
  - age at the end of a stint = age at start + its laps; a stint with age 0 at start opens a new set; a used one goes
    to the set of the same compound with the largest age not above its starting age (ties: most recently used).
Sessions in chronological order (Session key), so Sprint Qualifying is included (missing in our FastF1 cache).
Validation: every race stint on a used dry tyre with age >= 3 at start should match a pre-race set whose age equals it.
    python inventory_openf1_20261021.py        (from this folder)
"""
import unicodedata
import numpy as np
import pandas as pd

O = pd.read_excel('../../data/f1_2024_tire_usage.xlsx', 'OpenF1 Stints', header=3)
O['Driver'] = O['Piloto'].map(lambda n: {'Andrea Kimi Antonelli': 'ANT'}.get(
    n, unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().split()[-1][:3].upper()))
O = O.rename(columns={'Gran Premio': 'GP', 'Sesión': 'Session', 'N.º stint': 'Stint', 'Compuesto': 'Compound',
                      'Edad al inicio (vueltas)': 'Age0', 'Vuelta inicial': 'L0', 'Vuelta final': 'L1',
                      'Session key': 'SKey', 'Ronda': 'Round'})
O = O.sort_values(['Round', 'Driver', 'SKey', 'Stint'])
nxt = O.groupby(['Round', 'Driver', 'SKey']).L0.shift(-1)
O['Laps'] = np.where(nxt.notna(), nxt - O.L0, O.L1 - O.L0 + 1)
O['Age1'] = O.Age0 + O.Laps
DRY = ['SOFT', 'MEDIUM', 'HARD']
sets_rows, val_rows = [], []
for (rnd, gp, drv), g in O.groupby(['Round', 'GP', 'Driver']):
    sets = []
    for s in g[g.Session != 'Race'].itertuples():
        if s.Compound not in DRY:
            continue
        fresh = s.Age0 == 0
        cand = [x for x in sets if x['compound'] == s.Compound and x['life'] <= s.Age0] if not fresh else []
        if cand:
            best = max(cand, key=lambda x: (x['life'], x['last']))
        else:
            best = dict(compound=s.Compound, sid=len(sets) + 1, life=0.0, first=s.Session,
                        status='new' if fresh else 'unmatched', uses=[], last=0)
            sets.append(best)
        end = s.Age1 if np.isfinite(s.Age1) else s.Age0
        best['life'] = max(best['life'], end); best['last'] = s.SKey
        best['uses'].append(f"{s.Session}: {int(s.Age0)}-{int(end)}")
    for x in sets:
        sets_rows.append(dict(Round=rnd, GP=gp, Driver=drv, Set=x['sid'], Compound=x['compound'], Opened=x['first'],
                              Origin=x['status'], LapsBeforeRace=int(x['life']), Uses=' | '.join(x['uses'])))
    for s in g[(g.Session == 'Race') & g.Compound.isin(DRY) & (g.Age0 >= 3)].itertuples():
        lives = [x['life'] for x in sets if x['compound'] == s.Compound]
        val_rows.append(dict(Round=rnd, GP=gp, Driver=drv, RaceStint=s.Stint, StartLap=s.L0, Compound=s.Compound,
                             Age0=int(s.Age0), Exact=s.Age0 in lives,
                             Within1=any(abs(s.Age0 - v) <= 1 for v in lives), PreRaceLives=tuple(sorted(lives))))
S = pd.DataFrame(sets_rows); V = pd.DataFrame(val_rows)
S.to_csv('inventory_openf1_2024.csv', index=False); V.to_csv('inventory_openf1_2024_validation.csv', index=False)
print(f"{len(S)} juegos en {S.GP.nunique()} GP; origen: {S.Origin.value_counts().to_dict()}")
first = V.StartLap == 1
print(f"validacion (tandas de carrera con juego usado, edad >= 3): {len(V)}; exacto {V.Exact.mean():.2f}, +-1 {V.Within1.mean():.2f}")
print(f"  partida (vuelta 1): n={first.sum()}, exacto {V[first].Exact.mean():.2f}; "
      f"tras pits: n={(~first).sum()}, exacto {V[~first].Exact.mean():.2f}")
print(V[~V.Exact][['GP', 'Driver', 'RaceStint', 'StartLap', 'Compound', 'Age0', 'PreRaceLives']].to_string())
