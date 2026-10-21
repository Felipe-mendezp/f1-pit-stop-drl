"""Drivers who reached Q3 in each 2024 GP (FastF1 cache, offline): Q3 time present in the qualifying results -> q3_2024.csv."""
import fastf1, pandas as pd, logging
from pathlib import Path
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache(str(Path.home() / '.cache' / 'fastf1')); fastf1.Cache.offline_mode(True)
rows = []
for rnd in range(1, 25):
    s = fastf1.get_session(2024, rnd, 'Qualifying'); s.load(laps=False, telemetry=False, weather=False, messages=False)
    r = s.results
    for d, q3, pos in zip(r.Abbreviation, r.Q3, r.Position):
        rows.append(dict(Round=rnd, GP=s.event['EventName'], Driver=d, Q3=bool(pd.notna(q3)), QPos=pos))
    print(rnd, s.event['EventName'], int(r.Q3.notna().sum()), flush=True)
pd.DataFrame(rows).to_csv('q3_2024.csv', index=False)
