"""Descarga el estado final de TimingAppData y DriverList de cada carrera desde el espejo del live timing de F1
(livetiming-mirror.fastf1.dev, el mismo origen que usa FastF1) — 2026-10-28.

OpenF1 empieza en 2023, asi que para 2021-2022 esta es la unica fuente por tanda disponible. No es independiente del
origen de FastF1, pero si de su procesamiento (asignacion de compuesto por vuelta), que es donde estaba el error de
Belgica 2024. 2023 se descarga como control (OpenF1 ya mostro que solo Canada TSU difiere).
    python -I results/compound_check_livetiming_20261028/fetch_livetiming_20261028.py 2021 2022 2023
Salida: raw/<anio>/<GP>/{TimingAppData,DriverList}.json, raw/<anio>/index.json
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

BASE = 'https://livetiming-mirror.fastf1.dev/static/'
HERE = os.path.dirname(os.path.abspath(__file__))


def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(url)


def main(years):
    for y in years:
        out = os.path.join(HERE, 'raw', str(y))
        os.makedirs(out, exist_ok=True)
        idx = get(f'{BASE}{y}/Index.json')
        open(os.path.join(out, 'index.json'), 'wb').write(idx)
        meetings = json.loads(idx.decode('utf-8-sig'))['Meetings']
        for m in meetings:
            for s in m['Sessions']:
                if s.get('Type') != 'Race' or s.get('Name') != 'Race' or not s.get('Path'):
                    continue
                d = os.path.join(out, m['Name'].replace('/', '_'))
                os.makedirs(d, exist_ok=True)
                for f in ('TimingAppData.json', 'DriverList.json'):
                    p = os.path.join(d, f)
                    if os.path.exists(p):
                        continue
                    b = get(BASE + urllib.parse.quote(s["Path"]) + f)
                    if b is None:
                        print('  404', y, m['Name'], f); continue
                    open(p, 'wb').write(b)
                print(y, m['Name'], flush=True)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]])
