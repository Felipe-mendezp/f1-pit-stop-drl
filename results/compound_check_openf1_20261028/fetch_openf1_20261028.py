"""2026-10-28 — Descarga de OpenF1 (api.openf1.org) las tandas (stints) y pilotos de las carreras de un anio, para cotejar
el compuesto por vuelta de data/session_<anio>_V2.csv (FastF1). Salida: raw/<anio>/{meetings,sessions}.json y
raw/<anio>/{stints,drivers}_<session_key>.json. No se re-descarga lo que ya existe.
    python results/compound_check_openf1_20261028/fetch_openf1_20261028.py 2023
"""
import json, os, sys, time, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__))
Y = int(sys.argv[1])
OUT = os.path.join(HERE, 'raw', str(Y)); os.makedirs(OUT, exist_ok=True)


def get(url, path):
    if not os.path.exists(path):
        for k in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                json.loads(data)
                open(path, 'wb').write(data)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:   # sesion sin datos (p. ej. Imola 2023, cancelada)
                    open(path, 'w').write('[]'); break
                print('reintento', url, e); time.sleep(3 * (k + 1))
            except Exception as e:
                print('reintento', url, e); time.sleep(3 * (k + 1))
        time.sleep(0.4)
    return json.load(open(path))


get(f'https://api.openf1.org/v1/meetings?year={Y}', os.path.join(OUT, 'meetings.json'))
S = get(f'https://api.openf1.org/v1/sessions?year={Y}&session_name=Race', os.path.join(OUT, 'sessions.json'))
for s in S:
    k = s['session_key']
    st = get(f'https://api.openf1.org/v1/stints?session_key={k}', os.path.join(OUT, f'stints_{k}.json'))
    dr = get(f'https://api.openf1.org/v1/drivers?session_key={k}', os.path.join(OUT, f'drivers_{k}.json'))
    print(k, s['location'], len(st), 'tandas', len(dr), 'pilotos', flush=True)
