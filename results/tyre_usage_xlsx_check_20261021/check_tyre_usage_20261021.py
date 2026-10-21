"""Check data/f1_2024_tire_usage.xlsx (Lapchive sets + OpenF1 stints) against our FastF1 data.

  A. OpenF1 stints vs FastF1 stints (8 GPs of the paper, every session in laps_2024_8gp.csv).
  B. OpenF1 race stints vs data/session_2024_V2.csv (all 2024 races).
  C. Lapchive sets vs our reconstructed inventory (inventory_2024_8gp.csv): sets used before the race per
     driver x GP x compound and their pre-race laps.
  D. Race stints on a used tyre (OpenF1, all GPs) vs the Lapchive set that carries them into the race.
    python check_tyre_usage_20261021.py        (from this folder)
"""
import re
import unicodedata
import numpy as np
import pandas as pd

X = '../../data/f1_2024_tire_usage.xlsx'
INV = '../tyre_inventory_2024_20261021/'
L = pd.read_excel(X, 'Lapchive Sets', header=3)
O = pd.read_excel(X, 'OpenF1 Stints', header=3)


def abbr(name):
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return {'Andrea Kimi Antonelli': 'ANT'}.get(name, s.split()[-1][:3].upper())


for t in (L, O):
    t['Driver'] = t['Piloto'].map(abbr)
    t['GP'] = t['Gran Premio']
O['Compound'] = O['Compuesto']
O['Session'] = O['Sesión']
O['Stint'] = O['N.º stint']
O['Age0'] = O['Edad al inicio (vueltas)']
O['NLaps'] = O['Vuelta final'] - O['Vuelta inicial'] + 1
out = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s); out.append(s)


# ---------------------------------------------------------------- A. OpenF1 vs FastF1 stints (8 GP)
f = pd.read_csv(INV + 'laps_2024_8gp.csv')
F = (f.groupby(['GP', 'Driver', 'Session', 'Stint'])
     .agg(CompoundF=('Compound', 'first'), TL0=('TyreLife', 'first'), NF=('LapNumber', 'size'),
          Lap0=('LapNumber', 'first')).reset_index())
o8 = O[O.GP.isin(F.GP.unique()) & O.Session.isin(F.Session.unique())]
A = o8.merge(F, on=['GP', 'Driver', 'Session', 'Stint'], how='outer', indicator=True)
say('== A. OpenF1 vs FastF1 stints, 8 GP, sesiones FP1-3/Sprint/Q/Carrera')
say('emparejamiento:', A['_merge'].value_counts().to_dict())
b = A[A['_merge'] == 'both']
say(f"compuesto igual: {(b.Compound == b.CompoundF).mean():.4f} (n={len(b)})")
say(f"edad inicial OpenF1 == TyreLife FastF1 - 1: {(b.Age0 == b.TL0 - 1).mean():.4f}")
say(f"vuelta inicial igual: {(b['Vuelta inicial'] == b.Lap0).mean():.4f}")
nl = b.dropna(subset=['NLaps'])
say(f"vueltas OpenF1 (final-inicial+1) == FastF1: {(nl.NLaps == nl.NF).mean():.3f}; == FastF1+1: {(nl.NLaps == nl.NF + 1).mean():.3f}")
A.to_csv('A_openf1_vs_fastf1_stints.csv', index=False)

# ---------------------------------------------------------------- B. OpenF1 race stints vs session_2024_V2
r = pd.read_csv('../../data/session_2024_V2.csv').sort_values(['GP', 'Driver', 'LapNumber'])
r['new'] = (r.LapNumber == 1) | (r.groupby(['GP', 'Driver']).PitIn.shift(1) == 1)
r['Stint'] = r.groupby(['GP', 'Driver']).new.cumsum()
R = (r.groupby(['GP', 'Driver', 'Stint'])
     .agg(CompoundF=('Compound', 'first'), TL0=('TyreLife', 'first'), Lap0=('LapNumber', 'first')).reset_index())
orc = O[O.Session == 'Race']
B = orc.merge(R, on=['GP', 'Driver', 'Stint'], how='outer', indicator=True)
say('\n== B. Carrera: OpenF1 vs session_2024_V2 (24 GP)')
say('emparejamiento:', B['_merge'].value_counts().to_dict())
b = B[B['_merge'] == 'both']
say(f"compuesto igual: {(b.Compound == b.CompoundF).mean():.4f} (n={len(b)}); "
    f"edad inicial == TyreLife-1: {(b.Age0 == b.TL0 - 1).mean():.4f}; vuelta inicial igual: {(b['Vuelta inicial'] == b.Lap0).mean():.4f}")
B.to_csv('B_openf1_vs_session2024_race.csv', index=False)

# ---------------------------------------------------------------- Lapchive: parse per-session laps
SES = {'FP1': 'FP1', 'FP2': 'FP2', 'FP3': 'FP3', 'Quali': 'Q', 'Sprint': 'S', 'Race': 'R'}


def parse(det):
    det = str(det).split('|')[0]
    return {SES[k]: int(v) for k, v in re.findall(r'(FP1|FP2|FP3|Quali|Sprint|Race)\s+(\d+)', det)}


L['laps'] = L['Detalle de sesión'].map(parse)
L['LapsRace'] = L.laps.map(lambda d: d.get('R', 0))
L['LapsPre'] = L.laps.map(lambda d: sum(v for k, v in d.items() if k != 'R'))
L['Total'] = L['Vueltas acumuladas en el set']
L['Returned'] = L['Detalle de sesión'].astype(str).str.contains('return')
L['Merged'] = L['Vueltas estimadas'] == 'Sí'
L['Compound'] = L['Compuesto'].str.upper()
u = L[L.Estado == 'Used']
say('\n== Lapchive, consistencia interna')
say(f"sets usados: {len(u)}; total == suma por sesion: {(u.Total == u.LapsPre + u.LapsRace).mean():.3f}; "
    f"total > suma: {(u.Total > u.LapsPre + u.LapsRace).mean():.3f}; marcados 'estimado' (fusionados): {u.Merged.mean():.3f}")
say('asignacion por piloto-GP (H,M,S), frecuencias:',
    L[L.Compound.isin(['HARD', 'MEDIUM', 'SOFT'])].groupby(['GP', 'Driver', 'Compound']).size().unstack(fill_value=0)
    .apply(tuple, axis=1).value_counts().head(6).to_dict())

# ---------------------------------------------------------------- C. Lapchive sets vs our inventory (8 GP)
I = pd.read_csv(INV + 'inventory_2024_8gp.csv')
lp = u[u.GP.isin(I.GP.unique()) & u.Compound.isin(['HARD', 'MEDIUM', 'SOFT']) & (u.LapsPre > 0)]
key = ['GP', 'Driver', 'Compound']
cl = lp.groupby(key).LapsPre.agg(lambda s: tuple(sorted(s))).rename('Lapchive')
ct = lp.groupby(key).Total.agg(lambda s: tuple(sorted(s))).rename('LapchiveTotal')
ci = I.groupby(key).LapsBeforeRace.agg(lambda s: tuple(sorted(s))).rename('FastF1')
C = pd.concat([cl, ct, ci], axis=1).reset_index()
C['nL'] = C.Lapchive.map(lambda t: len(t) if isinstance(t, tuple) else 0)
C['nF'] = C.FastF1.map(lambda t: len(t) if isinstance(t, tuple) else 0)
C['same_n'] = C.nL == C.nF
C['same_laps'] = C.Lapchive == C.FastF1
C['laps_diff'] = [np.nan if not (isinstance(a, tuple) and isinstance(b, tuple)) else sum(a) - sum(b)
                  for a, b in zip(C.Lapchive, C.FastF1)]
say('\n== C. Sets usados antes de la carrera, Lapchive vs inventario FastF1 (8 GP, piloto x compuesto)')
say(f"celdas: {len(C)}; mismo n.º de sets: {C.same_n.mean():.3f}; mismas vueltas set a set: {C.same_laps.mean():.3f}; "
    f"misma suma de vueltas: {(C.laps_diff == 0).mean():.3f}")
say(C.groupby('GP')[['same_n', 'same_laps']].mean().round(2).to_string())
say(C.groupby('Compound')[['same_n', 'same_laps']].mean().round(2).to_string())
C.to_csv('C_lapchive_vs_inventory_8gp.csv', index=False)

# ---------------------------------------------------------------- D. race stints on used tyres vs Lapchive
ou = orc[(orc.Age0 > 0) & orc.Compound.isin(['HARD', 'MEDIUM', 'SOFT']) & orc.GP.isin(L.GP.unique())]
rows = []
for _, s in ou.iterrows():
    c = u[(u.GP == s.GP) & (u.Driver == s.Driver) & (u.Compound == s.Compound) & (u.LapsRace > 0) & (u.LapsPre > 0)]
    rows.append(dict(GP=s.GP, Driver=s.Driver, Stint=s.Stint, Compound=s.Compound, Age0=s.Age0,
                     n_sets=len(c), pre_match=bool((c.LapsPre == s.Age0).any()),
                     tot_match=bool(((c.Total - c.LapsRace) == s.Age0).any()),
                     pre=tuple(c.LapsPre), tot_minus_race=tuple(c.Total - c.LapsRace)))
D = pd.DataFrame(rows)
say('\n== D. Tandas de carrera con juego usado (OpenF1, 23 GP) vs set Lapchive que llega a la carrera')
say(f"tandas: {len(D)}; hay set Lapchive con uso previo y en carrera: {(D.n_sets > 0).mean():.3f}; "
    f"vueltas previas (suma por sesion) == edad OpenF1: {D.pre_match.mean():.3f}; "
    f"(total - carrera) == edad: {D.tot_match.mean():.3f}")
D.to_csv('D_race_used_stints_vs_lapchive.csv', index=False)
open('summary.txt', 'w').write('\n'.join(out) + '\n')
