"""2026-10-28 — Vueltas de las sesiones sprint 2021-2024, que extract_fpq_20261028.py no obtuvo (la cache local no las
tiene y la API de F1 responde 401). Fuentes:
  2021 sprint ("Sprint Qualifying" en el feed): FastF1 contra el mirror de live timing, con la ruta *_Sprint_Qualifying/
  2022 sprint: FastF1 contra el mirror (ruta estandar)
  2023 sprint shootout y sprint: OpenF1 (el mirror no tiene 2023)
  2024 sprint: data/practice_laps_2024_20260930.csv (FastF1, ya descargado); 2024 sprint qualifying: OpenF1
En OpenF1 no hay TrackStatus por vuelta (se deja 1) ni PitIn: se marca PitIn en la ultima vuelta de cada tanda salvo la
ultima tanda de la sprint (termina en la meta). TyreLife = tyre_age_at_start + (vuelta - lap_start) + 1, como FastF1.
Correr con el entorno de FastF1:  ~/.venvs/f1-fastf1/bin/python results/tyrelife_effective_20261028/extract_sprint_20261028.py
Salida: fpq_sprint_laps_2021_2024.csv.gz (mismas columnas que fpq_laps_2021_2024.csv.gz + Source)
"""
import os, json, time, unicodedata, warnings, logging, urllib.request
from pathlib import Path
import pandas as pd
import fastf1
warnings.filterwarnings('ignore'); logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache(str(Path.home() / '.cache' / 'fastf1'))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
COLS = ['Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound', 'TyreLife', 'FreshTyre', 'PitIn', 'PitOut',
        'TrackStatus', 'Deleted', 'Year', 'GP', 'Session', 'Source']
LOC = {'Baku': 'Azerbaijan Grand Prix', 'Spielberg': 'Austrian Grand Prix', 'Spa-Francorchamps': 'Belgian Grand Prix',
       'Lusail': 'Qatar Grand Prix', 'Austin': 'United States Grand Prix', 'São Paulo': 'São Paulo Grand Prix',
       'Shanghai': 'Chinese Grand Prix', 'Miami': 'Miami Grand Prix'}
nfc = lambda s: unicodedata.normalize('NFC', s)


def fastf1_sprint(y):
    out = []
    sch = fastf1.get_event_schedule(y, include_testing=False)
    for _, ev in sch.iterrows():
        if 'Sprint' not in [ev.get(f'Session{i}') for i in range(1, 6)]:
            continue
        s = fastf1.get_session(y, ev.EventName, 'Sprint')
        if y == 2021:
            s.api_path = s.api_path.replace('_Sprint/', '_Sprint_Qualifying/')
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        L = s.laps.copy()
        L['LapTime'] = L.LapTime.dt.total_seconds()
        L['PitIn'] = L.PitInTime.notna().astype(int); L['PitOut'] = L.PitOutTime.notna().astype(int)
        L['Year'], L['GP'], L['Session'], L['Source'] = y, nfc(ev.EventName), 'Sprint', 'fastf1-mirror'
        out.append(L[COLS]); print(y, ev.EventName, 'Sprint', len(L), flush=True)
    return out


def get(url):
    for k in range(5):
        try:
            time.sleep(0.4)
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except Exception:
            time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def openf1(y, names):
    out = []
    for ss in get(f'https://api.openf1.org/v1/sessions?year={y}'):
        if ss['session_name'] not in names:
            continue
        k, gp = ss['session_key'], LOC[nfc(ss['location'])]
        sess = 'Sprint' if ss['session_name'] == 'Sprint' else ('Sprint Shootout' if y == 2023 else 'Sprint Qualifying')
        drv = {d['driver_number']: d['name_acronym'] for d in get(f'https://api.openf1.org/v1/drivers?session_key={k}')}
        st = pd.DataFrame(get(f'https://api.openf1.org/v1/stints?session_key={k}'))
        la = pd.DataFrame(get(f'https://api.openf1.org/v1/laps?session_key={k}'))
        bad = st.lap_start.isna() | st.lap_end.isna() | st.tyre_age_at_start.isna()
        if bad.any():
            print('  ', y, gp, sess, 'tandas sin vueltas o edad en OpenF1 (se omiten):', int(bad.sum()))
        st = st[~bad]
        rows = []
        for _, r in st.iterrows():
            last = r.stint_number == st[st.driver_number == r.driver_number].stint_number.max()
            for n in range(int(r.lap_start), int(r.lap_end) + 1):
                rows.append(dict(driver_number=r.driver_number, lap_number=n, Stint=r.stint_number,
                                 Compound=r.compound, TyreLife=r.tyre_age_at_start + n - r.lap_start + 1,
                                 FreshTyre=r.tyre_age_at_start == 0,
                                 PitIn=int(n == r.lap_end and not (sess == 'Sprint' and last))))
        L = pd.DataFrame(rows).merge(la[['driver_number', 'lap_number', 'lap_duration', 'is_pit_out_lap']],
                                     on=['driver_number', 'lap_number'], how='left')
        L['Driver'] = L.driver_number.map(drv); L['LapNumber'] = L.lap_number
        L['LapTime'] = L.lap_duration; L['PitOut'] = L.is_pit_out_lap.fillna(False).astype(int)
        L['TrackStatus'], L['Deleted'] = '1', None
        L['Year'], L['GP'], L['Session'], L['Source'] = y, gp, sess, 'openf1'
        out.append(L[COLS]); print(y, gp, sess, len(L), 'sin tiempo', int(L.LapTime.isna().sum()), flush=True)
    return out


def main():
    out = fastf1_sprint(2021) + fastf1_sprint(2022)
    out += openf1(2023, ('Sprint Qualifying', 'Sprint')) + openf1(2024, ('Sprint Qualifying',))
    P = pd.read_csv(os.path.join(ROOT, 'data', 'practice_laps_2024_20260930.csv'))
    P = P[P.Session == 'Sprint'].copy(); P['Source'] = 'practice_laps_2024_20260930'; P['GP'] = P.GP.map(nfc)
    out.append(P[COLS]); print(2024, 'Sprint', len(P), P.GP.nunique(), 'GP')
    D = pd.concat(out, ignore_index=True)
    D.to_csv(os.path.join(HERE, 'fpq_sprint_laps_2021_2024.csv.gz'), index=False)
    print(D.groupby(['Year', 'Session']).agg(vueltas=('LapNumber', 'size'), GP=('GP', 'nunique')))


if __name__ == '__main__':
    main()
