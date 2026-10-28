"""2026-10-28 — Correcciones de compuesto en data/session_{2023,2024}_V2.csv (FastF1), cotejadas con OpenF1
(results/compound_check_openf1_20261028/; en 2024 tambien con Lapchive). Se aplican al leer el CSV (envuelve
pandas.read_csv), sin modificar los archivos:
  2024 Belgica  ALO 14-44, STR 13-44, TSU 16-44, RIC 22-44: FastF1 MEDIUM/SOFT -> HARD (C2)  [OpenF1 y Lapchive: HARD]
  2023 Canada   TSU 36-70: compuesto y vida vacios en FastF1 -> HARD (C3), vida 35..69 (sigue el juego de la vuelta 35,
                como dicen la vida de FastF1 en la vuelta 35 y OpenF1, que no registra cambio de juego)
2021-2022: sin fuente independiente (OpenF1 empieza en 2023).
"""
import os
import pandas as pd

_orig = pd.read_csv
BEL = {'ALO': (14, 44), 'STR': (13, 44), 'TSU': (16, 44), 'RIC': (22, 44)}


def _fix(df, name):
    if not {'GP', 'Driver', 'LapNumber', 'Compound'} <= set(df.columns):
        return df
    df = df.copy()
    n = 0
    if name == 'session_2024_V2.csv':
        for d, (a, b) in BEL.items():
            m = (df.GP == 'Belgian Grand Prix') & (df.Driver == d) & df.LapNumber.between(a, b)
            df.loc[m, 'Compound'] = 'HARD'
            if 'Compound_Detail' in df.columns:
                df.loc[m, 'Compound_Detail'] = 'C2'
            n += int(m.sum())
    elif name == 'session_2023_V2.csv':
        m = (df.GP == 'Canadian Grand Prix') & (df.Driver == 'TSU') & df.LapNumber.between(36, 70)
        assert df.loc[m, 'Compound'].isna().all(), 'Canada 2023 TSU ya tiene compuesto'
        df.loc[m, 'Compound'] = 'HARD'
        if 'Compound_Detail' in df.columns:
            df.loc[m, 'Compound_Detail'] = 'C3'
        if 'TyreLife' in df.columns:
            df.loc[m, 'TyreLife'] = df.loc[m, 'LapNumber'] - 1
        n += int(m.sum())
    print(f'[data_fix_20261028] {name}: {n} vueltas corregidas', flush=True)
    return df


def read_csv(path, *a, **k):
    df = _orig(path, *a, **k)
    name = os.path.basename(str(path))
    if name in ('session_2023_V2.csv', 'session_2024_V2.csv') and isinstance(df, pd.DataFrame):
        df = _fix(df, name)
    return df


def install():
    pd.read_csv = read_csv


install()
