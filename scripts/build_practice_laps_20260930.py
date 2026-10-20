"""
Lap-by-lap 2024 practice (and sprint) data for tyre-degradation estimation (2026-09-30).

Downloads, with FastF1, every session held BEFORE the 2024 Grand Prix race that
can inform tyre degradation: FP1-FP3 and, on sprint weekends, the Sprint race.
Qualifying / sprint qualifying are skipped (single push laps on fresh tyres).

Output: data/practice_laps_2024_20260930.csv (one row per lap; nothing is
filtered here, so the estimation script can apply and document its own filters).
Each lap carries the nearest weather sample (Rainfall, TrackTemp, AirTemp).

Run with the separate FastF1 venv (the pinned f1-drl-v2 env has no fastf1):
    ~/.venvs/f1-fastf1/bin/python scripts/build_practice_laps_20260930.py
The FastF1 cache lives outside the repo (~/.cache/fastf1).
"""

import os
import logging
import warnings
from pathlib import Path

import fastf1 as ff1
import pandas as pd

warnings.filterwarnings('ignore')
logging.getLogger('fastf1').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'practice_laps_2024_20260930.csv'
CACHE = Path.home() / '.cache' / 'fastf1'

SESSIONS = ['Practice 1', 'Practice 2', 'Practice 3', 'Sprint']

# Same 2024 mapping as scripts/build_dataframes.py (COMPOUND_MAPS[2024])
COMPOUND_MAP_2024 = {
    'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'São Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
}

LAP_COLS = ['Driver', 'Team', 'LapTime', 'LapNumber', 'Stint', 'Compound', 'TyreLife',
            'FreshTyre', 'PitInTime', 'PitOutTime', 'TrackStatus', 'IsAccurate',
            'Deleted', 'Time', 'LapStartTime']


def session_laps(year, gp, name):
    s = ff1.get_session(year, gp, name)
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = s.laps[[c for c in LAP_COLS if c in s.laps.columns]].copy()
    for c in ['LapTime', 'PitInTime', 'PitOutTime', 'Time', 'LapStartTime']:
        laps[c] = pd.to_timedelta(laps[c]).dt.total_seconds()
    w = s.weather_data
    if w is not None and len(w):
        w = w[['Time', 'Rainfall', 'TrackTemp', 'AirTemp']].copy()
        w['Time'] = pd.to_timedelta(w['Time']).dt.total_seconds()
        laps = pd.merge_asof(laps.sort_values('Time').dropna(subset=['Time']),
                             w.sort_values('Time'), on='Time', direction='nearest')
    laps['Session'] = name
    laps['GP'] = gp
    laps['Year'] = year
    laps['Compound_Detail'] = laps['Compound'].map(COMPOUND_MAP_2024.get(gp, {}))
    laps['PitIn'] = laps['PitInTime'].notna().astype(int)
    laps['PitOut'] = laps['PitOutTime'].notna().astype(int)
    return laps


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(CACHE))
    sched = ff1.get_event_schedule(2024, include_testing=False)
    frames = []
    for _, ev in sched.iterrows():
        gp = ev['EventName']
        names = [ev[f'Session{i}'] for i in range(1, 6)]
        for name in [n for n in names if n in SESSIONS]:
            try:
                df = session_laps(2024, gp, name)
                frames.append(df)
                print(f'{gp:28s} {name:11s} {len(df):5d} laps', flush=True)
            except Exception as e:
                print(f'{gp:28s} {name:11s} ERROR {e}', flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f'Saved {len(out)} laps to {OUT}')


if __name__ == '__main__':
    main()
