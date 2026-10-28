"""2026-10-28 — v10d: correcciones de compuesto de FastF1 aplicadas al leer data/session_{2022,2023,2024}_V2.csv (envuelve
pandas.read_csv; los archivos no se modifican). = data_fix_20261028 (v10c) + las 3 de 2022.
  2024 Belgica  ALO 14-44, STR 13-44, TSU 16-44, RIC 22-44: MEDIUM/SOFT -> HARD (C2)  [OpenF1 y Lapchive]
  2023 Canada   TSU 36-70: compuesto y vida vacios -> HARD (C3), vida = vuelta - 1  [OpenF1]
  2022 Belgica  GAS 23-44, TSU 31-44: SOFT (C4) -> MEDIUM (C3)  [live timing; Pirelli/Pitpass]
  2022 Monaco   LAT 3-19: HARD -> INTERMEDIATE (Compound_Detail vacio, como las demas intermedias);
                LAT 20-22: HARD con vida 18-20 -> vida 1-3 (parada real en la 19)  [live timing; FIA; RaceFans]
Detalle: results/compound_check_openf1_20261028/ y results/compound_check_livetiming_20261028/.
"""
import os
import numpy as np
import pandas as pd

_orig = pd.read_csv
BEL24 = {'ALO': (14, 44), 'STR': (13, 44), 'TSU': (16, 44), 'RIC': (22, 44)}
BEL22 = {'GAS': (23, 44), 'TSU': (31, 44)}


def _sel(df, gp, d, a, b):
    return (df.GP == gp) & (df.Driver == d) & df.LapNumber.between(a, b)


def _set(df, m, comp, detail):
    df.loc[m, 'Compound'] = comp
    if 'Compound_Detail' in df.columns:
        df.loc[m, 'Compound_Detail'] = detail


def _fix(df, name):
    if not {'GP', 'Driver', 'LapNumber', 'Compound'} <= set(df.columns):
        return df
    df = df.copy()
    n = 0
    if name == 'session_2024_V2.csv':
        for d, (a, b) in BEL24.items():
            m = _sel(df, 'Belgian Grand Prix', d, a, b)
            _set(df, m, 'HARD', 'C2'); n += int(m.sum())
    elif name == 'session_2023_V2.csv':
        m = _sel(df, 'Canadian Grand Prix', 'TSU', 36, 70)
        assert df.loc[m, 'Compound'].isna().all(), 'Canada 2023 TSU ya tiene compuesto'
        _set(df, m, 'HARD', 'C3')
        if 'TyreLife' in df.columns:
            df.loc[m, 'TyreLife'] = df.loc[m, 'LapNumber'] - 1
        n += int(m.sum())
    elif name == 'session_2022_V2.csv':
        for d, (a, b) in BEL22.items():
            m = _sel(df, 'Belgian Grand Prix', d, a, b)
            assert (df.loc[m, 'Compound'] == 'SOFT').all(), ('Belgica 2022', d)
            _set(df, m, 'MEDIUM', 'C3'); n += int(m.sum())
        m = _sel(df, 'Monaco Grand Prix', 'LAT', 3, 19)
        assert (df.loc[m, 'Compound'] == 'HARD').all(), 'Monaco 2022 LAT 3-19'
        _set(df, m, 'INTERMEDIATE', np.nan); n += int(m.sum())
        m = _sel(df, 'Monaco Grand Prix', 'LAT', 20, 22)
        if 'TyreLife' in df.columns:
            assert list(df.loc[m, 'TyreLife']) == [18.0, 19.0, 20.0], 'Monaco 2022 LAT 20-22'
            df.loc[m, 'TyreLife'] = df.loc[m, 'LapNumber'] - 19
        n += int(m.sum())
    print(f'[data_fix_v10d_20261028] {name}: {n} vueltas corregidas', flush=True)
    return df


def read_csv(path, *a, **k):
    df = _orig(path, *a, **k)
    name = os.path.basename(str(path))
    if name in ('session_2022_V2.csv', 'session_2023_V2.csv', 'session_2024_V2.csv') and isinstance(df, pd.DataFrame):
        df = _fix(df, name)
    return df


def install():
    pd.read_csv = read_csv


install()
