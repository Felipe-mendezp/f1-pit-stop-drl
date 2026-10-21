"""Qualifying (24 GP, FastF1 cache, offline) and Sprint Qualifying (6 sprint weekends, online attempt; on 21-oct the F1 API still fails, so none) -> q_sq_2024.csv."""
import fastf1, pandas as pd, logging
from pathlib import Path
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache(str(Path.home() / '.cache' / 'fastf1'))
rows = []
for rnd in range(1, 25):
    for name in ['Qualifying', 'Sprint Qualifying']:
        try:
            fastf1.Cache.offline_mode(name == 'Qualifying')
            s = fastf1.get_session(2024, rnd, name)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            L = s.laps[['Driver', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'FreshTyre']].copy()
            L['Session'] = name; L['GP'] = s.event['EventName']; L['Round'] = rnd; rows.append(L)
            print(rnd, s.event['EventName'], name, len(L), flush=True)
        except Exception as e:
            print('falta', rnd, name, type(e).__name__, str(e)[:80], flush=True)
pd.concat(rows).to_csv('q_sq_2024.csv', index=False)
