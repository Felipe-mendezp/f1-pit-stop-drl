"""2026-10-28 — v10eGQ: edad efectiva de los neumaticos (results/tyrelife_effective_20261028/, variante TL_GQ).
Se instala despues de data_fix_v10d_20261028 (compuestos corregidos) y reemplaza TyreLife de data/session_<anio>_V2.csv al
leerlo (los CSV no se modifican):
  G  : no cuentan las vueltas de carrera con SC, VSC o bandera roja
  GQ : ademas, la edad con que llega un juego usado cuenta solo vueltas de empuje en practicas y clasificacion
Esta carpeta usa TL_GQ. raw_read_csv lee con las correcciones de compuesto pero con la edad de FastF1 (para detectar los
pasos por el pit lane: TyreLife + 1 sin cambio de juego).
"""
import os
import pandas as pd
import data_fix_v10d_20261028  # noqa: F401  instala la correccion de compuestos

VARIANT = 'TL_GQ'
raw_read_csv = pd.read_csv
_AGES = None
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'tyrelife_effective_20261028', 'ages_2021_2024.csv.gz')


def _ages():
    global _AGES
    if _AGES is None:
        _AGES = raw_read_csv(_SRC)[['Year', 'GP', 'Driver', 'LapNumber', 'TL_raw', VARIANT]]
    return _AGES


def read_csv(path, *a, **k):
    df = raw_read_csv(path, *a, **k)
    name = os.path.basename(str(path))
    if name.startswith('session_20') and name.endswith('_V2.csv') and isinstance(df, pd.DataFrame)             and 'TyreLife' in df.columns:
        y = int(name[8:12])
        A = _ages()[lambda x: x.Year == y].drop(columns='Year')
        m = df[['GP', 'Driver', 'LapNumber']].merge(A, on=['GP', 'Driver', 'LapNumber'], how='left')
        ok = m[VARIANT].notna().values
        assert (m.TL_raw.values[ok] == df.TyreLife.values[ok]).all(), name
        assert df.TyreLife[~ok].isna().all(), name
        df = df.copy()
        df.loc[ok, 'TyreLife'] = m[VARIANT].values[ok]
        print(f'[age_fix_v10e {VARIANT}] {name}: edad cambiada en {(m[VARIANT].values[ok] != m.TL_raw.values[ok]).sum()} '
              f'vueltas', flush=True)
    return df


pd.read_csv = read_csv
