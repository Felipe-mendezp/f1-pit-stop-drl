"""Tyre sets available at the race start (opt-in, 2026-10-21).

F1_TYRE_SETS = 1 (the default file below) or the path of a CSV with columns GP, Driver, Compound, Age (one row per set
available at the start of the 2024 race; Age = laps before the race, new sets 0; Compound = SOFT/MEDIUM/HARD).
Built by scripts/build_tyre_sets_20261021.py (results/tyre_inventory_mixed_2024_20261021/). Unset = the environments
behave exactly as before (every stint starts on a new tyre and the supply is unlimited).
With the flag:
  - every driver starts the race with his real 2024 sets; each stint (start or pit stop) takes the least-worn set of
    the chosen compound, and its tyre life starts at the age of that set;
  - rivals with their real 2024 strategy (evaluation, Fig. 7) take the real age of each stint
    (data/tyre_race_stint_ages_2024_20261021.csv);
  - simulated rivals only choose compounds with sets left; the agent is penalised (F1_NO_SET_PENALTY, default -1000)
    and does not stop when it asks for a compound with no set left;
  - the agent observes 'tyre_sets' = [sets left per compound, age of the least-worn set per compound] (compounds in
    action order, ages 0 when none is left).
F1_TYRE_REUSE = 1 (with F1_TYRE_SETS; v9r2, author's idea of 2026-10-21, results/onedriver_v9r2_20261021/README.md):
  - the set taken off at a stop goes back to the driver's sets with its age (age when fitted + laps driven on it, the
    in-lap included unless under SC/VSC) and can be fitted again; a stop fits the least-worn set of the compound,
    which can be the one just taken off. Every driver has >= 1 set of each compound in the regulation inventory, so
    a stop never lacks a set;
  - the agent observes 'tyre_sets' = [age / 20 of the set it would fit by stopping now, per compound] (no counts:
    the number of sets left in the observation broke the learning of the single-driver agents in v9r).
The QMIP is unchanged: it plans with the sets in the pool (returned ones included, not the mounted one); an exact DP
over stints gives the same optimum with and without reuse in the 8 GPs (explore/dp_reuse_20261021.py).
"""
import bisect
import os
from functools import lru_cache

import pandas as pd

import config

# 2026-10-21 (evening): returns of the 2024 Sporting Regulations (art. 30.2), adopted by the author; the previous
# inventory (Lapchive's returns, 9.3 sets per driver) is data/tyre_sets_2024_20261021.csv (F1_TYRE_SETS=<path>)
DEFAULT_SETS = os.path.join(config.DATA_DIR, 'tyre_sets_2024_reg_20261021.csv')
DEFAULT_STINTS = os.path.join(config.DATA_DIR, 'tyre_race_stint_ages_2024_20261021.csv')
# Single-driver benchmark (generic driver): the sets of the multi-driver agent of each GP (v8 drivers)
ONE_DRIVER_SETS_OF = {
    'Bahrain Grand Prix': 'Driver_ALO', 'Saudi Arabian Grand Prix': 'Driver_NOR', 'Miami Grand Prix': 'Driver_OCO',
    'Emilia Romagna Grand Prix': 'Driver_TSU', 'Hungarian Grand Prix': 'Driver_TSU', 'Belgian Grand Prix': 'Driver_GAS',
    'Dutch Grand Prix': 'Driver_STR', 'Singapore Grand Prix': 'Driver_HUL', 'United States Grand Prix': 'Driver_MAG',
}
MAX_AGE = 60  # upper bound of a set's age before the race (2024 max: 39)


def enabled():
    return bool(os.environ.get('F1_TYRE_SETS'))


def reuse_enabled():
    return enabled() and os.environ.get('F1_TYRE_REUSE', '') not in ('', '0')


REUSE_AGE_SCALE = 20.0


def no_set_penalty():
    return float(os.environ.get('F1_NO_SET_PENALTY', '-1000'))


def _path():
    v = os.environ.get('F1_TYRE_SETS', '')
    return DEFAULT_SETS if v in ('', '1') else v


def compound_numbers(available_compounds):
    """{'HARD': c_min, 'MEDIUM': c_mid, 'SOFT': c_max} for the GP's three compounds (C1 hardest ... C5 softest)."""
    c = sorted(available_compounds)
    return {'HARD': c[0], 'MEDIUM': c[1], 'SOFT': c[2]}


@lru_cache(maxsize=4)
def _table(path):
    return pd.read_csv(path)


def sets_for(gp, driver, available_compounds):
    """{compound number: ascending list of set ages} of a driver ('Driver_XXX') at the start of the race."""
    t = _table(_path())
    t = t[(t.GP == gp) & (t.Driver == driver)]
    num = compound_numbers(available_compounds)
    out = {c: [] for c in available_compounds}
    for comp, age in zip(t.Compound, t.Age):
        out[num[comp]].append(int(age))
    for c in out:
        out[c].sort()
    return out


@lru_cache(maxsize=16)
def race_stint_ages(gp):
    """{driver: {pit lap (0 = start): age of the set fitted}} of the real 2024 race."""
    t = _table(DEFAULT_STINTS)
    t = t[t.GP == gp]
    out = {}
    for d, lap, age in zip(t.Driver, t.PitLap, t.Age):
        out.setdefault(d, {})[int(lap)] = int(age)
    return out


def take(sets, compound):
    """Pops and returns the age of the least-worn set of the compound (None when there is none left)."""
    lst = sets.get(compound)
    if not lst:
        return None
    return lst.pop(0)


def observation(sets, compounds):
    """[n sets left per compound] + [age of the least-worn set per compound (0 if none)], compounds in action order."""
    return [float(len(sets.get(c, []))) for c in compounds] + \
           [float(sets[c][0]) if sets.get(c) else 0.0 for c in compounds]


def give_back(sets, compound, age):
    """Reuse (F1_TYRE_REUSE): the set taken off goes back to the sets of its compound with its age."""
    bisect.insort(sets.setdefault(compound, []), int(age))


def has_set(sets, compound, mounted=None):
    """A set of the compound can be fitted (with reuse, the mounted compound always can: its own set comes back)."""
    return bool(sets.get(compound)) or (mounted is not None and compound == mounted[0])


def observation_reuse(sets, compounds, mounted):
    """Reuse: [age of the set that a stop now would fit, per compound] / REUSE_AGE_SCALE (0 if none); mounted =
    (compound, age it will have after this lap) of the set on the car."""
    out = []
    for c in compounds:
        cand = list(sets.get(c, [])[:1]) + ([mounted[1]] if c == mounted[0] else [])
        out.append(min(cand) / REUSE_AGE_SCALE if cand else 0.0)
    return out
