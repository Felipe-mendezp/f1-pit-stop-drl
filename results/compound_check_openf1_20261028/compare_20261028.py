"""2026-10-28 — Compuesto por vuelta de carrera: data/session_<anio>_V2.csv (FastF1) frente a OpenF1 (raw/<anio>/, de
fetch_openf1_20261028.py). Se excluyen las vueltas de PitIn (OpenF1 asigna la vuelta de entrada a la tanda siguiente) y
las de lluvia en cualquiera de las dos fuentes. Salida: mismatch_<anio>.csv (vueltas distintas) y resumen por tanda.
    python results/compound_check_openf1_20261028/compare_20261028.py 2023
"""
import json, os, sys, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
Y = int(sys.argv[1]); RAW = os.path.join(HERE, 'raw', str(Y))
meet = {m['meeting_key']: m['meeting_name'] for m in json.load(open(os.path.join(RAW, 'meetings.json')))}
rows = []
for s in json.load(open(os.path.join(RAW, 'sessions.json'))):
    k = s['session_key']; gp = meet[s['meeting_key']]
    drv = {d['driver_number']: d['name_acronym'] for d in json.load(open(os.path.join(RAW, f'drivers_{k}.json')))}
    for t in json.load(open(os.path.join(RAW, f'stints_{k}.json'))):
        if t.get('lap_start') is None or t.get('lap_end') is None:
            continue
        for lap in range(t['lap_start'], t['lap_end'] + 1):
            rows.append((gp, drv.get(t['driver_number'], str(t['driver_number'])), lap, str(t['compound']).upper(),
                         t['stint_number']))
O = pd.DataFrame(rows, columns=['GP', 'Driver', 'LapNumber', 'Compound_openf1', 'stint_openf1'])
O = O.sort_values('stint_openf1').drop_duplicates(['GP', 'Driver', 'LapNumber'], keep='last')
s = pd.read_csv(os.path.join(HERE, '..', '..', 'data', f'session_{Y}_V2.csv'))
gps_s, gps_o = set(s.GP), set(O.GP)
print('GP sin par en OpenF1:', sorted(gps_s - gps_o), '| OpenF1 sin par:', sorted(gps_o - gps_s))
m = s.merge(O, on=['GP', 'Driver', 'LapNumber'], how='left')
print(f'vueltas {len(m)}, sin par {m.Compound_openf1.isna().sum()}')
m = m[m.Compound_openf1.notna() & (m.PitIn != 1)]
WET = {'INTERMEDIATE', 'WET'}
m = m[~m.Compound.isin(WET) & ~m.Compound_openf1.isin(WET)]
bad = m[m.Compound != m.Compound_openf1].copy()
bad.to_csv(os.path.join(HERE, f'mismatch_{Y}.csv'), index=False)
print(f'vueltas secas comparadas {len(m)}, distintas {len(bad)} ({len(bad) / max(len(m), 1):.2%})')
if len(bad):
    T = bad.groupby(['GP', 'Driver', 'stint_openf1']).agg(desde=('LapNumber', 'min'), hasta=('LapNumber', 'max'),
                                                          n=('LapNumber', 'size'), fastf1=('Compound', 'first'),
                                                          openf1=('Compound_openf1', 'first')).reset_index()
    T.to_csv(os.path.join(HERE, f'mismatch_stints_{Y}.csv'), index=False)
    pd.set_option('display.width', 250); print(T.to_string(index=False))
