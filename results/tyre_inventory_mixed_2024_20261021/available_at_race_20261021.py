"""Tyre sets available at the start of each 2024 race (24 GPs), from the mixed inventory and Lapchive's returns.

Input: inventory_mixed_2024.csv (inventory_mixed_20261021.py) and the 'Lapchive Sets' sheet of
data/f1_2024_tire_usage.xlsx (columns 'Detalle de sesión' = per-session laps and '↩ FPx return', 'Retorno estimado').
  - Weekends without sprint: Lapchive marks the sets returned after FP1/FP2/FP3 (an estimate from the regulations,
    not an observation). For each driver x GP x compound x session, the returned sets are matched to our sets whose
    last pre-race use is that session: same count -> all of them are returned ('direct'); more candidates -> the k
    sets closest in laps to Lapchive's (ties: the most worn) ('ambiguous'); fewer candidates -> all candidates are
    returned and the missing returns stay unassigned ('short'). Lapchive's laps are only used to pick sets; the laps
    kept are ours (Lapchive adds 1 lap per stint, see ../tyre_usage_xlsx_check_20261021/README.md).
  - Sprint weekends (Lapchive marks no return): our rule, sets whose last use is FP1 are returned; sets used in
    Sprint Qualifying, Sprint or Qualifying stay.
  - New sets at the race = allocation - sets used before the race; allocation (H, M, S) = (2, 3, 8) on normal weekends
    and (2, 4, 6) on sprint ones (exactly what Lapchive lists for every driver-weekend; Las Vegas assumed normal).
Checks against the race (data/session_2024_V2.csv): race stints on a used dry set must find an available set of that
age; fresh race stints per compound must not exceed the new sets.
Outputs: available_at_race_2024.csv (one row per available set, new ones with 0 laps), available_at_race_2024_summary.csv
(driver x GP x compound), returns_matching_2024.csv, available_summary.txt.
    python available_at_race_20261021.py [K]    (from this folder)
K (optional, default 0): sets with K laps or fewer before the race are never returned (installation or scrub laps);
with K > 0 the outputs get the suffix _keepK.
"""
import itertools
import sys
import unicodedata
import numpy as np
import pandas as pd

DRY = ['SOFT', 'MEDIUM', 'HARD']
FP = {'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3'}


def abbr(n):
    return {'Andrea Kimi Antonelli': 'ANT'}.get(n, unicodedata.normalize('NFKD', n).encode('ascii', 'ignore')
                                                  .decode().split()[-1][:3].upper())


S = pd.read_csv('inventory_mixed_2024.csv')

# Substitutes (2026-10-21): one tyre allocation per car, so a driver outside the simulator's 20 (FP1 rookie or race
# substitute) is merged into the regular 2024 driver whose car he drove: same team and no session in common (FP1
# rookies, BEA in Saudi Arabia), or the regular driver absent from the whole weekend (COL for SAR, LAW for RIC, BEA for
# MAG, DOO for OCO). Applied to our sets, Lapchive's sets and the race laps.
TEAM = {'ALB': 'Williams', 'SAR': 'Williams', 'ALO': 'Aston Martin', 'STR': 'Aston Martin', 'BOT': 'Kick Sauber',
        'ZHO': 'Kick Sauber', 'GAS': 'Alpine', 'OCO': 'Alpine', 'HAM': 'Mercedes', 'RUS': 'Mercedes',
        'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team', 'LEC': 'Ferrari', 'SAI': 'Ferrari', 'NOR': 'McLaren',
        'PIA': 'McLaren', 'PER': 'Red Bull Racing', 'VER': 'Red Bull Racing', 'RIC': 'RB', 'TSU': 'RB'}
_pt = (pd.read_csv('../../data/practice_laps_2024_20260930.csv', usecols=['GP', 'Driver', 'Team'], low_memory=False)
       .drop_duplicates(['GP', 'Driver']).set_index(['GP', 'Driver']).Team)
_ses = S.assign(ses=S.Uses.str.findall(r'(Practice \d|Sprint Qualifying|Sprint|Qualifying)'))
RENAME = {}
for gp, g in _ses.groupby('GP'):
    ses = {d: set(sum(x.ses.tolist(), [])) for d, x in g.groupby('Driver')}
    for x in sorted(set(ses) - set(TEAM)):
        mates = [d for d in TEAM if TEAM[d] == _pt.get((gp, x))]
        r_ = [d for d in mates if d not in ses] or [d for d in mates if not (ses[d] & ses[x])]
        assert len(r_) == 1, (gp, x, r_)
        RENAME[(gp, x)] = r_[0]


def rename(df):
    df['Driver'] = [RENAME.get((g, d), d) for g, d in zip(df.GP, df.Driver)]
    return df


S = rename(S)
S['LastSession'] = S.Uses.str.split(' | ', regex=False).str[-1].str.split(':').str[0]
S['Returned'] = False
S['ReturnMatch'] = ''
L = pd.read_excel('../../data/f1_2024_tire_usage.xlsx', 'Lapchive Sets', header=3)
L['Driver'] = L.Piloto.map(abbr); L['Compound'] = L.Compuesto.str.upper(); L['GP'] = L['Gran Premio']
L = rename(L[L.Compound.isin(DRY)].copy())
L['RetAfter'] = L['Detalle de sesión'].astype(str).str.extract(r'↩ (FP\d) return')[0].map(FP)
sprint = L.groupby('GP')['Sesión(es)'].apply(lambda s: s.astype(str).str.contains('SPRINT').any())
sprint = sprint.reindex(S.GP.unique()).fillna(False)
sprint['Las Vegas Grand Prix'] = False  # not in Lapchive; normal weekend
# Lapchive allocation (used + never fitted) is (H, M, S) = (2, 3, 8) on all 353 normal driver-weekends and (2, 4, 6)
# on all 120 sprint ones; the rule is applied to every driver x GP (also Las Vegas, missing in Lapchive)
ALLOC = {False: {'HARD': 2, 'MEDIUM': 3, 'SOFT': 8}, True: {'HARD': 2, 'MEDIUM': 4, 'SOFT': 6}}
alloc = pd.Series({(gp, d, c): ALLOC[bool(sprint[gp])][c] for gp, d in S[['GP', 'Driver']].drop_duplicates()
                   .itertuples(index=False) for c in DRY}, name='Allocation')
alloc.index.names = ['GP', 'Driver', 'Compound']

match_rows = []
for (gp, drv, comp, ses), g in L[L.RetAfter.notna()].groupby(['GP', 'Driver', 'Compound', 'RetAfter']):
    idx = S.index[(S.GP == gp) & (S.Driver == drv) & (S.Compound == comp) & (S.LastSession == ses) & ~S.Returned]
    k, n = len(g), len(idx)
    lap_lc = sorted(g['Vueltas acumuladas en el set'].tolist())
    if n <= k:
        chosen, kind = list(idx), 'direct' if n == k else 'short'
    else:
        best = None
        for comb in itertools.combinations(idx, k):
            ours = sorted(S.loc[list(comb), 'LapsBeforeRace'])
            key = (sum(abs(a - b) for a, b in zip(lap_lc, ours)), -sum(ours))
            if best is None or key < best[0]:
                best = (key, list(comb))
        chosen, kind = best[1], 'ambiguous'
    S.loc[chosen, 'Returned'] = True
    S.loc[chosen, 'ReturnMatch'] = kind
    match_rows.append(dict(GP=gp, Driver=drv, Compound=comp, Session=ses, LapchiveReturned=k, Candidates=n,
                           Kind=kind, LapchiveLaps=' '.join(str(int(x)) for x in lap_lc),
                           OurReturnedLaps=' '.join(str(x) for x in sorted(S.loc[chosen, 'LapsBeforeRace']))))
spr = S.GP.map(sprint).astype(bool)
fp1_only = spr & (S.LastSession == 'Practice 1')
S.loc[fp1_only, 'Returned'] = True
S.loc[fp1_only, 'ReturnMatch'] = 'sprint rule'
KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SUF = f'_keep{KEEP}' if KEEP else ''
short = S.Returned & (S.LapsBeforeRace <= KEEP)
S.loc[short, 'Returned'] = False
S.loc[short, 'ReturnMatch'] = S.loc[short, 'ReturnMatch'] + f' -> kept (<= {KEEP} laps)'
Mt = pd.DataFrame(match_rows)

# available sets: used and kept + new ones
A = S[~S.Returned][['Round', 'GP', 'Driver', 'Compound', 'Set', 'LapsBeforeRace', 'LastSession', 'Uses']].copy()
A['State'] = 'used'
used = S.groupby(['GP', 'Driver', 'Compound']).size().rename('SetsUsed')
cnt = pd.concat([alloc, used], axis=1).fillna(0)
cnt['New'] = (cnt.Allocation - cnt.SetsUsed).clip(lower=0).astype(int)
rounds = S.drop_duplicates('GP').set_index('GP').Round
new_rows = [dict(Round=rounds.get(gp), GP=gp, Driver=drv, Compound=c, Set=np.nan, LapsBeforeRace=0,
                 LastSession='', Uses='', State='new')
            for (gp, drv, c), r in cnt.iterrows() for _ in range(int(r.New)) if gp in rounds.index]
A = pd.concat([A, pd.DataFrame(new_rows)]).sort_values(['Round', 'Driver', 'Compound', 'LapsBeforeRace'])
A = A[A.Driver.isin(S.Driver.unique()) | (A.State == 'used')]

M = (A.groupby(['Round', 'GP', 'Driver', 'Compound'])
     .agg(New=('State', lambda s: int((s == 'new').sum())), UsedKept=('State', lambda s: int((s == 'used').sum())),
          UsedLaps=('LapsBeforeRace', lambda v: ' '.join(str(int(x)) for x in sorted(v) if x > 0)))
     .reset_index())
ret = S[S.Returned].groupby(['GP', 'Driver', 'Compound']).size().rename('Returned')
M = M.merge(ret.reset_index(), on=['GP', 'Driver', 'Compound'], how='left').fillna({'Returned': 0})
M = M.merge(cnt[['Allocation', 'SetsUsed']].reset_index(), on=['GP', 'Driver', 'Compound'], how='left')
M['Sprint'] = M.GP.map(sprint).astype(bool)
flags = S.groupby(['GP', 'Driver']).ReturnMatch.agg(lambda s: ','.join(sorted(set(x for x in s if x in ('ambiguous', 'short')))))
M['Flags'] = M.set_index(['GP', 'Driver']).index.map(flags).fillna('')
A.to_csv(f'available_at_race_2024{SUF}.csv', index=False)
M.to_csv(f'available_at_race_2024_summary{SUF}.csv', index=False)
Mt.to_csv(f'returns_matching_2024{SUF}.csv', index=False)

# ---------------------------------------------------------------- checks against the race
r = rename(pd.read_csv('../../data/session_2024_V2.csv', low_memory=False)).sort_values(['GP', 'Driver', 'LapNumber'])
r['Stint'] = r.groupby(['GP', 'Driver']).PitOut.cumsum() + 1
RS = (r.groupby(['GP', 'Driver', 'Stint']).agg(Compound=('Compound', 'first'), TL0=('TyreLife', 'first'),
                                              Lap0=('LapNumber', 'first')).reset_index())
RS = RS[RS.Compound.isin(DRY)]
RS['Age0'] = RS.TL0 - 1
# age of the set fitted at each real race stint (start: PitLap 0; after a stop: the PitIn lap), for the simulator's
# rivals with their real 2024 strategy
RS.assign(PitLap=(RS.Lap0 - 1).astype(int), Age=RS.Age0.astype(int))[['GP', 'Driver', 'PitLap', 'Compound', 'Age']] \
    .to_csv(f'race_stint_ages_2024{SUF}.csv', index=False)
out = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


say(f"devoluciones Lapchive emparejadas (grupos piloto x GP x compuesto x sesion): {len(Mt)}; "
    f"{Mt.Kind.value_counts().to_dict()}; devoluciones sin juego nuestro: "
    f"{int((Mt.LapchiveReturned - Mt.Candidates).clip(lower=0).sum())}")
say(f"juegos usados antes de la carrera: {len(S)}; devueltos: {int(S.Returned.sum())} "
    f"({S[S.Returned].ReturnMatch.value_counts().to_dict()}); disponibles usados: {int((~S.Returned).sum())}")
say('\npromedio por piloto y GP de juegos disponibles al inicio de la carrera:')
say(M.groupby(['Sprint', 'Compound'])[['New', 'UsedKept', 'Returned']].mean().round(2).to_string())
av = A[A.State == 'used'].groupby(['GP', 'Driver', 'Compound']).LapsBeforeRace.apply(list)
pre_all = S.groupby(['GP', 'Driver', 'Compound']).LapsBeforeRace.apply(list)
U = RS[(RS.Age0 >= 1)].copy()
U['InUsed'] = [a in pre_all.get((g, d, c), []) for g, d, c, a in zip(U.GP, U.Driver, U.Compound, U.Age0)]
U['InAvail'] = [a in av.get((g, d, c), []) for g, d, c, a in zip(U.GP, U.Driver, U.Compound, U.Age0)]
U = U[U.InUsed]  # race stints explained by a pre-race set (excludes in-race reuse)
say(f"\ntandas de carrera con juego usado que calzan con un juego previo: {len(U)}; ese juego sigue disponible "
    f"tras las devoluciones: {U.InAvail.mean():.3f} (edad >= 3: {U[U.Age0 >= 3].InAvail.mean():.3f}, "
    f"n={int((U.Age0 >= 3).sum())})")
say(U[~U.InAvail][['GP', 'Driver', 'Stint', 'Lap0', 'Compound', 'Age0']].to_string(index=False))
F = RS[RS.Age0 == 0].groupby(['GP', 'Driver', 'Compound']).size().rename('FreshRace').reset_index()
F = F.merge(M[['GP', 'Driver', 'Compound', 'New']], on=['GP', 'Driver', 'Compound'], how='left')
bad = F[F.FreshRace > F.New]
say(f"\ntandas de carrera con juego nuevo: {int(F.FreshRace.sum())}; celdas con mas juegos nuevos usados que "
    f"disponibles: {len(bad)} de {len(F)}")
say(bad.to_string(index=False))
RU = RS.groupby(['GP', 'Driver', 'Compound']).size().rename('RaceStints').reset_index()
RU = RU.merge(M[['GP', 'Driver', 'Compound', 'New', 'UsedKept']], on=['GP', 'Driver', 'Compound'], how='left').fillna(0)
zero = RU[(RU.New + RU.UsedKept) == 0]
say(f"\ncompuestos usados en carrera sin ningun juego disponible: {len(zero)} de {len(RU)}")
say(zero.to_string(index=False))
tot = M.groupby(['GP', 'Driver', 'Sprint'])[['New', 'UsedKept']].sum().sum(axis=1).groupby('Sprint').describe()
say('\njuegos disponibles por piloto y GP (los 3 compuestos):'); say(tot.round(2).to_string())
open(f'available_summary{SUF}.txt', 'w').write('\n'.join(out) + '\n')
