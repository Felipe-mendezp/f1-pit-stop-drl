import fastf1, pandas as pd, logging, sys
from pathlib import Path
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache(str(Path.home() / '.cache' / 'fastf1')); fastf1.Cache.offline_mode(True)
rows = []
for gp in ['Bahrain', 'Saudi Arabia', 'Miami', 'Hungary', 'Belgium', 'Netherlands', 'Singapore', 'United States']:
    for name in ['Qualifying', 'Sprint Qualifying', 'Race']:
        try:
            s = fastf1.get_session(2024, gp, name); s.load(laps=True, telemetry=False, weather=False, messages=False)
            L = s.laps[['Driver', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'FreshTyre']].copy()
            L['Session'] = name; L['GP'] = s.event['EventName']; rows.append(L); print(gp, name, len(L), flush=True)
        except Exception as e:
            print('falta', gp, name, type(e).__name__, flush=True)
pd.concat(rows).to_csv('quali_race_2024_8gp.csv', index=False)
