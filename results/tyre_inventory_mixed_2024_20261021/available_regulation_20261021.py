"""Tyre sets available at the start of each 2024 race with the returns of the 2024 Sporting Regulations — 2026-10-21.

Art. 30.2 (Issue 6, 30-apr-2024), dry-weather sets:
  normal weekend (13 sets): 2 returned after P1, 2 after P2, 2 after P3; a driver who reached Q3 also returns one set
    of the Q3 specification (soft) at covers-on after qualifying -> 7 sets at the race (6 for Q3 cars);
  sprint weekend (12 sets): 1 returned after P1, 1 after the sprint (the set with most laps in the sprint, if one was
    used there), 3 after qualifying -> 7 sets at the race;
  one set of each mandatory race specification (hard, medium) may not be returned before the race;
  normal weekend: one soft set may not be used nor returned before Q3 (a driver out of Q3 keeps it new).
Returns are unobserved, so which sets are returned is a hypothesis. At each return, the candidates are the sets not
used in any later session (and not protected); they are ranked by RULE:
  used sets first ('most' laps first, or 'least' laps first), then new sets of the compound given by NEW:
  'surplus' (compound with most new candidates; ties soft > medium > hard), 'soft', 'medium' or 'hard' first;
  or RULE = 'score' with weights (w_soft, w_medium): the set with the highest laps + w[compound] is returned first
  (w_hard = 0; ties: the compound with most candidates).
Protected: the least-worn hard and the least-worn medium. Sets used before the race and drivers as in
available_at_race_20261021.py (inventory_mixed_2024.csv, substitutes merged into the regular driver of the car).
Validation with the real races (race_stint_ages_2024_keep1.csv), stints in order: a stint on a new set takes a new set
of its compound; a stint on a used set must find an available set of that age (or one used earlier in the race).
    python available_regulation_20261021.py            # all variants, summary table
    python available_regulation_20261021.py most surplus  # one variant -> available_regulation_2024_<used>_<new>.csv
    python available_regulation_20261021.py score 20,5
"""
import sys
import itertools
import numpy as np
import pandas as pd

DRY = ['SOFT', 'MEDIUM', 'HARD']
NORMAL = {'HARD': 2, 'MEDIUM': 3, 'SOFT': 8}
SPRINT = {'HARD': 2, 'MEDIUM': 4, 'SOFT': 6}
TEAM = {'ALB': 'Williams', 'SAR': 'Williams', 'ALO': 'Aston Martin', 'STR': 'Aston Martin', 'BOT': 'Kick Sauber',
        'ZHO': 'Kick Sauber', 'GAS': 'Alpine', 'OCO': 'Alpine', 'HAM': 'Mercedes', 'RUS': 'Mercedes',
        'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team', 'LEC': 'Ferrari', 'SAI': 'Ferrari', 'NOR': 'McLaren',
        'PIA': 'McLaren', 'PER': 'Red Bull Racing', 'VER': 'Red Bull Racing', 'RIC': 'RB', 'TSU': 'RB'}
SES_RE = r'(Practice \d|Sprint Qualifying|Sprint|Qualifying)'

S = pd.read_csv('inventory_mixed_2024.csv')
S['uses'] = S.Uses.str.findall(r'(Practice \d|Sprint Qualifying|Sprint|Qualifying): (\d+)-(\d+)')
_pt = (pd.read_csv('../../data/practice_laps_2024_20260930.csv', usecols=['GP', 'Driver', 'Team'], low_memory=False)
       .drop_duplicates(['GP', 'Driver']).set_index(['GP', 'Driver']).Team)
RENAME = {}
for gp, g in S.groupby('GP'):
    ses = {d: {u[0] for us in x.uses for u in us} for d, x in g.groupby('Driver')}
    for x in sorted(set(ses) - set(TEAM)):
        mates = [d for d in TEAM if TEAM[d] == _pt.get((gp, x))]
        r_ = [d for d in mates if d not in ses] or [d for d in mates if not (ses[d] & ses[x])]
        assert len(r_) == 1, (gp, x, r_)
        RENAME[(gp, x)] = r_[0]
S['Driver'] = [RENAME.get((g, d), d) for g, d in zip(S.GP, S.Driver)]
SPR = S.groupby('GP').uses.apply(lambda us: any(u[0] == 'Sprint' for x in us for u in x))
SPR['Las Vegas Grand Prix'] = False
Q3 = pd.read_csv('q3_2024.csv')
Q3 = set(zip(Q3.GP, Q3.Driver)) & set(zip(Q3[Q3.QPos <= 10].GP, Q3[Q3.QPos <= 10].Driver))
Q3 = {(g, RENAME.get((g, d), d)) for g, d in Q3}
R = pd.read_csv('race_stint_ages_2024_keep1.csv')
NL = pd.read_csv('../../data/session_2024_V2.csv', usecols=['GP', 'LapNumber']).groupby('GP').LapNumber.max()
ORDER_N = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying']
ORDER_S = ['Practice 1', 'Sprint Qualifying', 'Sprint', 'Qualifying']


def build(gp, drv, used_rule, new_rule):
    """Sets at the race start [(compound, laps)] and the number of returns that found no candidate."""
    sprint = bool(SPR[gp])
    order = ORDER_S if sprint else ORDER_N
    g = S[(S.GP == gp) & (S.Driver == drv)]
    sets = []  # dict(comp, laps, last (index of last session used, -1 new), sprint_laps)
    for c, life, uses in zip(g.Compound, g.LapsBeforeRace, g.uses):
        idx = [order.index(u[0]) for u in uses if u[0] in order]
        sets.append(dict(comp=c, laps=int(life), last=max(idx) if idx else -1,
                         sprint_laps=sum(int(b) - int(a) for s_, a, b in uses if s_ == 'Sprint')))
    alloc = SPRINT if sprint else NORMAL
    for c in DRY:
        for _ in range(max(0, alloc[c] - sum(s['comp'] == c for s in sets))):
            sets.append(dict(comp=c, laps=0, last=-1, sprint_laps=0))
    for c in ('HARD', 'MEDIUM'):  # protected: least-worn set of each race specification
        cand = [s for s in sets if s['comp'] == c]
        if cand:
            min(cand, key=lambda s: s['laps'])['protected'] = True
    # normal weekend: one set of the Q3 specification (soft) may not be used nor returned before Q3 (art. 30.2 h i);
    # a driver who did not reach Q3 keeps it new for the race (Q3 cars return one soft after qualifying, below)
    if not sprint and (gp, drv) not in Q3:
        new_soft = [s for s in sets if s['comp'] == 'SOFT' and s['laps'] == 0]
        if new_soft:
            new_soft[0]['protected'] = True
    events = ([(0, 1, None), (2, 1, 'sprint'), (3, 3, None)] if sprint
              else [(0, 2, None), (1, 2, None), (2, 2, None)] + ([(3, 1, 'q3')] if (gp, drv) in Q3 else []))
    missing = 0
    for after, k, kind in events:
        for _ in range(k):
            cand = [s for s in sets if not s.get('returned') and not s.get('protected') and s['last'] <= after]
            if kind == 'q3':
                cand = [s for s in cand if s['comp'] == 'SOFT']
            if kind == 'sprint' and any(s['sprint_laps'] for s in cand):
                pick = max(cand, key=lambda s: s['sprint_laps'])
            elif used_rule == 'score':  # new_rule = (w_soft, w_medium): highest laps + w[compound] returned first
                w = {'SOFT': new_rule[0], 'MEDIUM': new_rule[1], 'HARD': 0.0}
                n = {c: sum(s['comp'] == c for s in cand) for c in DRY}
                if not cand:
                    missing += 1
                    continue
                pick = max(cand, key=lambda s: (s['laps'] + w[s['comp']], n[s['comp']], -DRY.index(s['comp'])))
            else:
                used = [s for s in cand if s['laps'] > 0]
                if used:
                    pick = (max if used_rule == 'most' else min)(used, key=lambda s: s['laps'])
                elif cand:
                    pref = {'soft': ['SOFT', 'MEDIUM', 'HARD'], 'medium': ['MEDIUM', 'SOFT', 'HARD'],
                            'hard': ['HARD', 'MEDIUM', 'SOFT']}
                    if new_rule == 'surplus':
                        n = {c: sum(s['comp'] == c for s in cand) for c in DRY}
                        comp = max(DRY, key=lambda c: (n[c], -DRY.index(c)))
                    else:
                        comp = next(c for c in pref[new_rule] if any(s['comp'] == c for s in cand))
                    pick = next(s for s in cand if s['comp'] == comp)
                else:
                    missing += 1
                    continue
            pick['returned'] = True
    return [(s['comp'], s['laps']) for s in sets if not s.get('returned')], missing


def validate(avail):
    """Race check: unexplained used stints, cells where new sets run out, compounds used without any set."""
    unexpl = short = nosets = n_used = 0
    bad_cells = set()
    for (gp, drv), g in R.groupby(['GP', 'Driver']):
        pool = {c: sorted(l for cc, l in avail.get((gp, drv), []) if cc == c) for c in DRY}
        race = {c: [] for c in DRY}  # lives of sets used in this race
        g = g.sort_values('PitLap').reset_index(drop=True)
        ends = list(g.PitLap[1:]) + [NL[gp]]
        for c in set(g.Compound):
            if not pool[c]:
                nosets += 1
        for (lap, c, age), end in zip(zip(g.PitLap, g.Compound, g.Age), ends):
            life_end = age + (end - lap)
            if age == 0:
                if 0 in pool[c]:
                    pool[c].remove(0)
                else:
                    short += 1; bad_cells.add((gp, drv, c))
            else:
                n_used += 1
                if age in pool[c]:
                    pool[c].remove(age)
                elif any(abs(age - x) <= 1 for x in race[c]):
                    race[c].remove(min(race[c], key=lambda x: abs(age - x)))
                else:
                    unexpl += 1
            race[c].append(life_end)
    return dict(used_stints=n_used, used_unexplained=unexpl, new_short=short, new_short_cells=len(bad_cells),
                compound_without_set=nosets)


def run(used_rule, new_rule):
    avail, miss = {}, 0
    for gp, drv in S[['GP', 'Driver']].drop_duplicates().itertuples(index=False):
        avail[(gp, drv)], m = build(gp, drv, used_rule, new_rule)
        miss += m
    return avail, miss


if __name__ == '__main__':
    cur = pd.read_csv('available_at_race_2024_keep1.csv')
    base = {(g, d): list(zip(x.Compound, x.LapsBeforeRace.astype(int))) for (g, d), x in cur.groupby(['GP', 'Driver'])}
    rows = [dict(variant='actual (Lapchive, K=1)', returns_without_candidate=np.nan,
                 sets_per_driver=np.mean([len(v) for v in base.values()]), **validate(base))]
    if len(sys.argv) > 2:
        variants = [(sys.argv[1], tuple(float(x) for x in sys.argv[2].split(',')) if sys.argv[1] == 'score'
                     else sys.argv[2])]
    else:
        variants = list(itertools.product(['most', 'least'], ['surplus', 'soft', 'medium', 'hard'])) + \
            [('score', (ws, wm)) for ws in (5, 10, 20, 40, 1000) for wm in (0, ws / 4, ws / 2)]
    for ur, nr in variants:
        avail, miss = run(ur, nr)
        rows.append(dict(variant=f'reglamento: usados {ur}, nuevos {nr}', returns_without_candidate=miss,
                         sets_per_driver=np.mean([len(v) for v in avail.values()]), **validate(avail)))
        if len(variants) == 1:
            pd.DataFrame([dict(GP=g, Driver=d, Compound=c, Age=l, Sprint=bool(SPR[g]), Q3=(g, d) in Q3)
                          for (g, d), v in avail.items() for c, l in v]).sort_values(['GP', 'Driver', 'Compound', 'Age']) \
                .to_csv(f'available_regulation_2024_{ur}_{"_".join(map(str, nr)) if isinstance(nr, tuple) else nr}.csv',
                        index=False)
    T = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(T.round(2).to_string(index=False))
    if len(variants) > 1:
        T.to_csv('available_regulation_variants.csv', index=False)
