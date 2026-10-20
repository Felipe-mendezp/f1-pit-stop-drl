"""
2026-10-10 — Mejor vuelta por piloto en FP1-FP3 y Qualifying, temporadas 2021-2024 (Ec. 2, corrección
de coeficientes de piloto). Una fila por (Year, GP, Session, Driver).

Columnas: Year, GP (EventName de FastF1), Round, Session, Driver, BestLapTime [s], n_laps,
wet_share (fraccion de vueltas de la sesion con INTERMEDIATE/WET).
Salida: data/session_bestlaps_2021_2024_20261010.csv (nada se sobrescribe: si existe, aborta).
Correr con el venv de FastF1 (el pinned f1-drl-v2 no tiene fastf1):
    ~/.venvs/f1-fastf1/bin/python scripts/build_session_bestlaps_20261010.py
Cache de FastF1 fuera del repo (~/.cache/fastf1). Reanudable: guarda parciales por año en ~/.cache/fastf1/bestlaps_<year>.csv
"""
import logging, time, warnings
from pathlib import Path
import fastf1 as ff1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

warnings.filterwarnings('ignore'); logging.getLogger('fastf1').setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'session_bestlaps_2021_2024_20261010.csv'
CACHE = Path.home() / '.cache' / 'fastf1'; CACHE.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(str(CACHE))
SESSIONS = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying']


def load_laps(year, rnd, sname, ename):
    """Vueltas de la sesion; None si no existe o no tiene datos. Ante el limite de la API
    (500 llamadas/h) espera y reintenta: nunca se registra una sesion como vacia por eso."""
    while True:
        try:
            s = ff1.get_session(year, rnd, sname)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            return s.laps
        except RateLimitExceededError:
            print(f'  limite de la API en {year} {ename} {sname}: espero 10 min', flush=True)
            time.sleep(600)
        except Exception as e:  # sesion inexistente (sprint) o sin datos
            print(f'  {year} {ename} {sname}: sin datos ({type(e).__name__})', flush=True)
            return None


def year_rows(year):
    part = CACHE / f'bestlaps_{year}.csv'
    if part.exists():
        return pd.read_csv(part)
    rows = []
    while True:
        try:
            sched = ff1.get_event_schedule(year, include_testing=False); break
        except RateLimitExceededError:
            print('  limite de la API (calendario): espero 10 min', flush=True); time.sleep(600)
    for _, ev in sched.iterrows():
        for sname in SESSIONS:
            laps = load_laps(year, ev['RoundNumber'], sname, ev['EventName'])
            if laps is None:
                continue
            if laps is None or len(laps) == 0:
                continue
            wet = float(laps['Compound'].isin(['INTERMEDIATE', 'WET']).mean())
            ok = laps[laps['LapTime'].notna()]
            if 'Deleted' in ok:
                ok = ok[ok['Deleted'] != True]
            for drv, g in ok.groupby('Driver'):
                rows.append(dict(Year=year, GP=ev['EventName'], Round=int(ev['RoundNumber']), Session=sname,
                                 Driver=drv, BestLapTime=g['LapTime'].min().total_seconds(),
                                 n_laps=len(g), wet_share=wet))
        print(year, ev['EventName'], 'ok', flush=True)
    df = pd.DataFrame(rows); df.to_csv(part, index=False)
    return df


def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} ya existe')
    df = pd.concat([year_rows(y) for y in (2021, 2022, 2023, 2024)], ignore_index=True)
    df.to_csv(OUT, index=False)
    print('->', OUT, len(df))


if __name__ == '__main__':
    main()
