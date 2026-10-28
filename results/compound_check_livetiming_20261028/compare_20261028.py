"""Compara el compuesto por vuelta de data/session_<anio>_V2.csv (FastF1) con las tandas del live timing (estado final
de TimingAppData, raw/<anio>/) — 2026-10-28.

Cada tanda del live timing tiene compuesto, StartLaps (edad al montar) y TotalLaps (edad al final); sus vueltas de
carrera son TotalLaps - StartLaps y se asignan en orden desde la vuelta 1. Se excluyen las vueltas de PitIn y PitOut
(la frontera depende de la convencion). Compuestos secos = SOFT/MEDIUM/HARD en ambas fuentes; las vueltas con compuesto
faltante o UNKNOWN en FastF1 se reportan aparte, con lo que dice el live timing.
    python -I results/compound_check_livetiming_20261028/compare_20261028.py 2021 2022
Salida: mismatch_<anio>.csv (vueltas), mismatch_stints_<anio>.csv (tramos contiguos), resumen en pantalla.
"""
import json, os, sys, unicodedata
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DRY = {'SOFT', 'MEDIUM', 'HARD'}


def live_laps(year):
    rows, lens = [], []
    base = os.path.join(HERE, 'raw', str(year))
    for gp in sorted(os.listdir(base)):
        p = os.path.join(base, gp, 'TimingAppData.json')
        if not os.path.exists(p):
            continue
        app = json.load(open(p, encoding='utf-8-sig'))['Lines']
        drv = json.load(open(os.path.join(base, gp, 'DriverList.json'), encoding='utf-8-sig'))
        for num, line in app.items():
            st = line.get('Stints') or []
            st = [st[k] for k in sorted(st, key=int)] if isinstance(st, dict) else st
            code = drv.get(num, {}).get('Tla', num)
            lap = 0
            for i, s in enumerate(st):
                n = int(s.get('TotalLaps', 0)) - int(s.get('StartLaps', 0))
                for k in range(n):
                    lap += 1
                    rows.append(dict(GP=unicodedata.normalize("NFC", gp), Driver=code, LapNumber=lap, live=s.get('Compound'), live_stint=i,
                                     live_age=int(s.get('StartLaps', 0)) + k + 1,
                                     live_tnc=s.get('TyresNotChanged')))
            lens.append(dict(GP=unicodedata.normalize("NFC", gp), Driver=code, live_laps=lap))
    return pd.DataFrame(rows), pd.DataFrame(lens)


def main(year):
    s = pd.read_csv(os.path.join(ROOT, 'data', f'session_{year}_V2.csv'))
    L, N = live_laps(year)
    gps = set(s.GP)
    print(f'== {year}: GP en datos {len(gps)}, en live timing {L.GP.nunique()}, sin live: {sorted(gps - set(L.GP))}')
    m = s.merge(L, on=['GP', 'Driver', 'LapNumber'], how='left')
    # largo de carrera por piloto: datos vs live timing
    nd = s.groupby(['GP', 'Driver']).LapNumber.max().rename('data_laps').reset_index().merge(N, how='left')
    off = nd[nd.data_laps != nd.live_laps]
    print(f'pilotos con distinto numero de vueltas (datos vs live): {len(off)} de {len(nd)}')
    if len(off):
        print(off.to_string(index=False))
    border = (m.PitIn == 1) | (m.PitOut == 1)
    dry = m.Compound.isin(DRY) & m.live.isin(DRY) & ~border
    bad = m[dry & (m.Compound != m.live)]
    print(f'vueltas secas comparadas {int(dry.sum())}, diferencias {len(bad)}')
    miss = m[~m.Compound.isin(DRY | {'INTERMEDIATE', 'WET'})]
    print(f'vueltas sin compuesto (NaN/UNKNOWN) en FastF1: {len(miss)}; el live timing dice:',
          miss.live.value_counts(dropna=False).to_dict())
    wet = m[(m.Compound.isin({'INTERMEDIATE', 'WET'}) | m.live.isin({'INTERMEDIATE', 'WET'})) & ~border
            & (m.Compound != m.live) & m.live.notna()]
    print(f'diferencias con compuestos de lluvia: {len(wet)}')
    out = pd.concat([bad.assign(kind='seco'), miss.assign(kind='faltante'), wet.assign(kind='lluvia')])
    out = out[['kind', 'GP', 'Driver', 'LapNumber', 'Compound', 'Compound_Detail', 'TyreLife', 'live', 'live_age',
               'live_stint', 'live_tnc', 'PitIn', 'PitOut']].sort_values(['kind', 'GP', 'Driver', 'LapNumber'])
    out.to_csv(os.path.join(HERE, f'mismatch_{year}.csv'), index=False)
    if len(out):
        g = out.assign(b=(out.LapNumber.diff() != 1) | (out.Driver != out.Driver.shift()) | (out.GP != out.GP.shift())
                       | (out.kind != out.kind.shift()))
        g['blk'] = g.b.cumsum()
        T = g.groupby('blk').agg(kind=('kind', 'first'), GP=('GP', 'first'), Driver=('Driver', 'first'),
                                 desde=('LapNumber', 'min'), hasta=('LapNumber', 'max'), n=('LapNumber', 'size'),
                                 fastf1=('Compound', lambda x: '/'.join(sorted(set(map(str, x))))),
                                 live=('live', lambda x: '/'.join(sorted(set(map(str, x))))))
        T.to_csv(os.path.join(HERE, f'mismatch_stints_{year}.csv'), index=False)
        print(T.to_string(index=False))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        main(int(a))


# --- Comparacion por secuencia de compuestos (robusta a largos de tanda mal registrados en el live timing, p. ej.
# Austria 2022, donde el feed deja de actualizar TotalLaps). Secuencias colapsadas: tandas consecutivas del mismo
# compuesto cuentan una vez. Un NaN/UNKNOWN de FastF1 se reemplaza por lo que diga el live timing en esa posicion.
def collapse(seq):
    out = []
    for c in seq:
        if not out or out[-1] != c:
            out.append(c)
    return out


def sequences(year):
    s = pd.read_csv(os.path.join(ROOT, 'data', f'session_{year}_V2.csv')).sort_values(['GP', 'Driver', 'LapNumber'])
    s['c'] = s.Compound.where(s.Compound.isin(DRY | {'INTERMEDIATE', 'WET'}), '?')
    base = os.path.join(HERE, 'raw', str(year))
    rows = []
    for gp in sorted(os.listdir(base)):
        p = os.path.join(base, gp, 'TimingAppData.json')
        if not os.path.exists(p):
            continue
        app = json.load(open(p, encoding='utf-8-sig'))['Lines']
        drv = json.load(open(os.path.join(base, gp, 'DriverList.json'), encoding='utf-8-sig'))
        g = unicodedata.normalize('NFC', gp)
        for num, line in app.items():
            st = line.get('Stints') or []
            st = [st[k] for k in sorted(st, key=int)] if isinstance(st, dict) else st
            code = drv.get(num, {}).get('Tla', num)
            live = collapse([x.get('Compound') for x in st])
            dd = s[(s.GP == g) & (s.Driver == code)]
            if dd.empty:
                continue
            data = collapse(list(dd.c))
            ok = len(data) == len(live) and all(a == b or a == '?' for a, b in zip(data, live))
            fill = [(a, b) for a, b in zip(data, live) if a == '?'] if len(data) == len(live) else []
            rows.append(dict(GP=g, Driver=code, data='-'.join(data), live='-'.join(map(str, live)), ok=ok,
                             filled=';'.join(f'?->{b}' for _, b in fill),
                             live_stints=' | '.join(f"{x.get('Compound')} {x.get('StartLaps')}->{x.get('TotalLaps')}"
                                                    f"{' tnc' if x.get('TyresNotChanged') == '1' else ''}" for x in st)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, f'sequences_{year}.csv'), index=False)
    bad = R[~R.ok]
    print(f'== {year} secuencias: {len(R)} pilotos, distintas {len(bad)}; NaN/UNKNOWN completados:',
          R[R.filled != ''][['GP', 'Driver', 'filled']].values.tolist())
    if len(bad):
        print(bad[['GP', 'Driver', 'data', 'live', 'live_stints']].to_string(index=False))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        sequences(int(a))
