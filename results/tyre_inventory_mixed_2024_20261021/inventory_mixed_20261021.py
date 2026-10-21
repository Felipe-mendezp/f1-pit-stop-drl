"""Pre-race tyre-set inventory, 24 GPs of 2024, mixing FastF1 and OpenF1 (run from this folder).

Sources per session (stints of dry compounds only):
  FP1-FP3, Sprint : data/practice_laps_2024_20260930.csv (FastF1)
  Qualifying      : q_sq_2024.csv (FastF1 cache, load_q_sq_20261021.py)
  Sprint Qualif.  : 'OpenF1 Stints' sheet of data/f1_2024_tire_usage.xlsx (not in our FastF1 cache; the F1 API
                    answers 401). OpenF1 counts the boundary lap twice, so a non-final stint has next_start - start
                    laps; TyreLife convention: first lap = age at start + 1.
  Race (check)    : data/session_2024_V2.csv (stint = 1 + cumulative PitOut).
Assignment rule (as in ../tyre_inventory_2024_20261021/inventory_20261021.py): a stint on a fresh tyre opens a set;
a used one goes to the set of the same compound whose life (age at its last use) is the largest one not above the
stint's starting age (ties: most recently used; if none, one lap of tolerance); if none fits, an 'unmatched' set is
created. Fresh = TyreLife at the first lap <= 1 (FastF1's FreshTyre flag is wrong on some stints).
Checks: (1) sets opened per compound vs the allocation in the 'Lapchive Sets' sheet (used + never fitted);
(2) every race stint that starts on a used dry tyre (age >= 3) must find a set (pre-race or earlier in the race) whose
life equals its starting age.
Outputs: inventory_mixed_2024.csv (one row per set), inventory_mixed_2024_summary.csv (driver x GP x compound),
inventory_mixed_2024_validation.csv, summary.txt.
"""
import unicodedata
import numpy as np
import pandas as pd

R = '../../data/'
X = R + 'f1_2024_tire_usage.xlsx'
DRY = ['SOFT', 'MEDIUM', 'HARD']
ORDER = {'Practice 1': 1, 'Practice 2': 2, 'Sprint Qualifying': 2, 'Practice 3': 3, 'Sprint': 3, 'Qualifying': 4}


def abbr(n):
    return {'Andrea Kimi Antonelli': 'ANT'}.get(n, unicodedata.normalize('NFKD', n).encode('ascii', 'ignore')
                                                  .decode().split()[-1][:3].upper())


def ff1_stints(d):
    d = d.dropna(subset=['TyreLife', 'Stint']).sort_values(['GP', 'Driver', 'Session', 'LapNumber'])
    g = d.groupby(['GP', 'Driver', 'Session', 'Stint'])
    s = g.agg(Compound=('Compound', 'first'), TL0=('TyreLife', 'first'), TL1=('TyreLife', 'max'),
              Fresh=('FreshTyre', 'first'), Lap0=('LapNumber', 'first')).reset_index()
    s['Fresh'] = s.TL0 <= 1  # FreshTyre is False on 90 stints that start at TyreLife 1 (68 in Mexico Q)
    s['Src'] = 'FastF1'
    return s


p = pd.read_csv(R + 'practice_laps_2024_20260930.csv', low_memory=False)
q = pd.read_csv('q_sq_2024.csv')
O = pd.read_excel(X, 'OpenF1 Stints', header=3)
O['Driver'] = O['Piloto'].map(abbr)
O = O.rename(columns={'Gran Premio': 'GP', 'Sesión': 'Session', 'N.º stint': 'Stint', 'Compuesto': 'Compound',
                      'Edad al inicio (vueltas)': 'Age0', 'Vuelta inicial': 'Lap0', 'Vuelta final': 'L1'})
sq = O[O.Session == 'Sprint Qualifying'].sort_values(['GP', 'Driver', 'Stint']).copy()
nxt = sq.groupby(['GP', 'Driver']).Lap0.shift(-1)
sq['Laps'] = np.where(nxt.notna(), nxt - sq.Lap0, sq.L1 - sq.Lap0 + 1)
sq['TL0'] = sq.Age0 + 1
sq['TL1'] = sq.Age0 + sq.Laps.fillna(1)
sq['Fresh'] = sq.Age0 == 0
sq['Src'] = 'OpenF1'
ST = pd.concat([ff1_stints(p), ff1_stints(q[q.Session == 'Qualifying']),
                sq[['GP', 'Driver', 'Session', 'Stint', 'Compound', 'TL0', 'TL1', 'Fresh', 'Lap0', 'Src']]])
ST = ST[ST.Compound.isin(DRY)]
ST['Order'] = ST.Session.map(ORDER)

r = pd.read_csv(R + 'session_2024_V2.csv', low_memory=False).sort_values(['GP', 'Driver', 'LapNumber'])
r['Stint'] = r.groupby(['GP', 'Driver']).PitOut.cumsum() + 1
r['Session'] = 'Race'; r['FreshTyre'] = r.TyreLife <= 1
RS = ff1_stints(r)
rounds = O.drop_duplicates('GP').set_index('GP')['Ronda']


def assign(sets, comp, tl0, tl1, fresh, ses, order):
    age0 = tl0 - 1
    cand = [x for x in sets if x['compound'] == comp and x['life'] <= age0] if not fresh else []
    if not fresh and not cand:  # TyreLife glitches: allow a set that is one lap older than the stint start
        cand = [x for x in sets if x['compound'] == comp and x['life'] <= age0 + 1]
    if cand:
        best = max(cand, key=lambda x: (x['life'], x['last']))
    else:
        best = dict(compound=comp, sid=len(sets) + 1, life=0.0, first=ses, uses=[], last=0,
                    status='new' if fresh else 'unmatched')
        sets.append(best)
    best['life'] = max(best['life'], tl1); best['last'] = order
    best['uses'].append(f"{ses}: {int(age0)}-{int(tl1)}")
    return best


set_rows, val_rows = [], []
for (gp, drv), g in ST.groupby(['GP', 'Driver']):
    sets = []
    for s in g.sort_values(['Order', 'Lap0']).itertuples():
        assign(sets, s.Compound, s.TL0, s.TL1, s.Fresh, s.Session, s.Order)
    for x in sets:
        set_rows.append(dict(Round=rounds.get(gp), GP=gp, Driver=drv, Compound=x['compound'], Set=x['sid'],
                             Opened=x['first'], Origin=x['status'], LapsBeforeRace=int(x['life']),
                             Uses=' | '.join(x['uses'])))
    rs = RS[(RS.GP == gp) & (RS.Driver == drv) & RS.Compound.isin(DRY)].sort_values('Lap0')
    for s in rs.itertuples():
        age0 = s.TL0 - 1
        if age0 >= 3:
            pre = [x['life'] for x in sets if x['compound'] == s.Compound and x['first'] != 'Race']
            race = [x['life'] for x in sets if x['compound'] == s.Compound and x['first'] == 'Race']
            val_rows.append(dict(Round=rounds.get(gp), GP=gp, Driver=drv, RaceStint=s.Stint, StartLap=int(s.Lap0),
                                 Compound=s.Compound, Age0=int(age0), PreRace=age0 in pre,
                                 InRace=age0 in race and age0 not in pre, PreRaceLives=tuple(sorted(pre))))
        assign(sets, s.Compound, s.TL0, s.TL1, s.Fresh, 'Race', 9)
S = pd.DataFrame(set_rows).sort_values(['Round', 'Driver', 'Compound', 'Set'])
V = pd.DataFrame(val_rows).sort_values(['Round', 'Driver', 'RaceStint'])

# allocation from Lapchive (used + never fitted), driver x GP x compound
L = pd.read_excel(X, 'Lapchive Sets', header=3)
L['Driver'] = L['Piloto'].map(abbr); L['Compound'] = L['Compuesto'].str.upper()
alloc = L[L.Compound.isin(DRY)].groupby(['Gran Premio', 'Driver', 'Compound']).size()
alloc.index.names = ['GP', 'Driver', 'Compound']
M = S.groupby(['Round', 'GP', 'Driver', 'Compound']).agg(
    SetsUsed=('Set', 'size'), Unmatched=('Origin', lambda o: int((o == 'unmatched').sum())),
    LapsBeforeRace=('LapsBeforeRace', lambda v: ' '.join(str(x) for x in sorted(v)))).reset_index()
M = M.merge(alloc.rename('Allocation').reset_index(), on=['GP', 'Driver', 'Compound'], how='outer')
M['Round'] = M.GP.map(rounds)
M['SetsUsed'] = M.SetsUsed.fillna(0).astype(int)
M['NewSets'] = M.Allocation - M.SetsUsed
M = M.sort_values(['Round', 'Driver', 'Compound'])
S.to_csv('inventory_mixed_2024.csv', index=False)
M.to_csv('inventory_mixed_2024_summary.csv', index=False)
V.to_csv('inventory_mixed_2024_validation.csv', index=False)

out = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


say(f"juegos usados antes de la carrera: {len(S)} en {S.GP.nunique()} GP; origen {S.Origin.value_counts().to_dict()}")
say('unmatched por GP:', S[S.Origin == 'unmatched'].GP.value_counts().to_dict())
m = M.dropna(subset=['Allocation'])
say(f"celdas piloto x GP x compuesto con asignacion Lapchive: {len(m)}; juegos usados > asignacion: {(m.NewSets < 0).sum()}")
say(m[m.NewSets < 0][['GP', 'Driver', 'Compound', 'SetsUsed', 'Allocation', 'LapsBeforeRace']].to_string(index=False))
say(f"\nvalidacion: {len(V)} tandas de carrera con juego usado (edad >= 3); juego previo a la carrera exacto "
    f"{V.PreRace.mean():.2f}; reutilizado dentro de la carrera {V.InRace.mean():.2f}; sin explicar "
    f"{(~V.PreRace & ~V.InRace).mean():.2f}")
first = V.StartLap == 1
say(f"  partida: n={first.sum()}, exacto {V[first].PreRace.mean():.2f}")
say(V[~V.PreRace & ~V.InRace][['GP', 'Driver', 'RaceStint', 'StartLap', 'Compound', 'Age0', 'PreRaceLives']]
    .to_string(index=False))
say('\nvueltas antes de la carrera por juego usado (por compuesto):')
say(S.groupby('Compound').LapsBeforeRace.describe().round(1).to_string())
open('summary.txt', 'w').write('\n'.join(out) + '\n')
