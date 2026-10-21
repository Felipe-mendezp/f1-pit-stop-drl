"""Tyre-set inputs of the simulator (opt-in F1_TYRE_SETS) — 2026-10-21.

Copies the validated inventory (results/tyre_inventory_mixed_2024_20261021/, threshold K = 1: sets with 1 lap before
the race are never returned; substitutes merged into the regular driver of their car) to data/:
  data/tyre_sets_2024_20261021.csv        GP, Driver, Compound, Age   one row per set available at the race start
                                          (Age = laps before the race; new sets Age 0)
  data/tyre_race_stint_ages_2024_20261021.csv  GP, Driver, PitLap, Compound, Age   age of the set fitted at each
                                          real 2024 race stint (PitLap 0 = start, else the PitIn lap, as GP_CONFIGS)
Drivers as in the simulator ('Driver_XXX'). Rebuild the inputs first:
    cd results/tyre_inventory_mixed_2024_20261021 && python inventory_mixed_20261021.py && python available_at_race_20261021.py 1
    python scripts/build_tyre_sets_20261021.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'tyre_inventory_mixed_2024_20261021')
A = pd.read_csv(os.path.join(SRC, 'available_at_race_2024_keep1.csv'))
A = A.assign(Driver='Driver_' + A.Driver, Age=A.LapsBeforeRace.astype(int))[['GP', 'Driver', 'Compound', 'Age']]
A = A.sort_values(['GP', 'Driver', 'Compound', 'Age']).reset_index(drop=True)
R = pd.read_csv(os.path.join(SRC, 'race_stint_ages_2024_keep1.csv'))
R = R.assign(Driver='Driver_' + R.Driver).sort_values(['GP', 'Driver', 'PitLap']).reset_index(drop=True)
A.to_csv(os.path.join(ROOT, 'data', 'tyre_sets_2024_20261021.csv'), index=False)
R.to_csv(os.path.join(ROOT, 'data', 'tyre_race_stint_ages_2024_20261021.csv'), index=False)
c = A.groupby(['GP', 'Driver', 'Compound']).size().unstack(fill_value=0)
print(f'{len(A)} juegos, {A.GP.nunique()} GP, {A.groupby("GP").Driver.nunique().min()}-{A.groupby("GP").Driver.nunique().max()} '
      f'pilotos por GP; minimo de juegos por compuesto: {c.min().to_dict()}; {len(R)} tandas de carrera')
