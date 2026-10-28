"""2026-10-28 — Vueltas de practicas y clasificacion 2021-2024 desde la cache local de FastF1 (offline), para saber cuantas
vueltas de cada juego fueron "de empuje" (vueltas lentas: salida, entrada, enfriamiento, bajo bandera, sin tiempo).
Correr con el entorno de FastF1:  ~/.venvs/f1-fastf1/bin/python results/tyrelife_effective_20261028/extract_fpq_20261028.py
Salida: fpq_laps_2021_2024.csv.gz (una fila por vuelta)
"""
import os, warnings, logging
from pathlib import Path
import pandas as pd
import fastf1
warnings.filterwarnings('ignore'); logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache(str(Path.home() / '.cache' / 'fastf1')); fastf1.Cache.offline_mode(True)
HERE = os.path.dirname(os.path.abspath(__file__))
SESS = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Sprint Qualifying', 'Sprint Shootout', 'Sprint']
COLS = ['Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound', 'TyreLife', 'FreshTyre', 'PitInTime', 'PitOutTime',
        'TrackStatus', 'Deleted']
out = []
for y in (2021, 2022, 2023, 2024):
    sch = fastf1.get_event_schedule(y, include_testing=False)
    for _, ev in sch.iterrows():
        for s in SESS:
            if s not in [ev.get(f'Session{i}') for i in range(1, 6)]:
                continue
            try:
                ss = fastf1.get_session(y, ev.EventName, s)
                ss.load(laps=True, telemetry=False, weather=False, messages=False)
                L = ss.laps[[c for c in COLS if c in ss.laps.columns]].copy()
            except Exception as e:
                print('  sin datos', y, ev.EventName, s, type(e).__name__); continue
            L['LapTime'] = L.LapTime.dt.total_seconds()
            L['PitIn'] = L.PitInTime.notna().astype(int); L['PitOut'] = L.PitOutTime.notna().astype(int)
            L = L.drop(columns=['PitInTime', 'PitOutTime'])
            L['Year'], L['GP'], L['Session'] = y, ev.EventName, s
            out.append(L)
            print(y, ev.EventName, s, len(L), flush=True)
pd.concat(out, ignore_index=True).to_csv(os.path.join(HERE, 'fpq_laps_2021_2024.csv.gz'), index=False)
