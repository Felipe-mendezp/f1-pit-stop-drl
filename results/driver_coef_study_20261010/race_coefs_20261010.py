"""2026-10-10 — Coeficientes de piloto por carrera (GP-anio), 2021-2024, para la Ec. 2.

Correr desde la raiz del repo: python results/driver_coef_study_20261010/race_coefs_20261010.py
Salida: results/driver_coef_study_20261010/race_coefs.csv

Para cada edicion (Year, GP) se ajusta la Ec. 1 SOLO con las vueltas de esa carrera, con el mismo
procedimiento que el simulador (models.regressions: transformaciones de all_laps, filtro de outliers
+15 s, IQR iterativo, referencia VER). Ediciones con >10 % de vueltas WET/INT fuera (filtro A).
Columnas: Year, GP, Driver, coef (s/vuelta vs VER), se, n_laps (vueltas limpias del piloto),
lap_ref (mediana de LapTime en pista limpia sin PitIn/PitOut de la carrera, s).
Nota: data/coef_drivers_2024.csv NO sirve como verdad por carrera: incluye pilotos que no corrieron
en 2024 (VET, GIO, ...), es decir, viene de una regresion con anios anteriores.
"""
import io, os, sys, contextlib
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R

OUT = os.path.dirname(os.path.abspath(__file__))


def prep(raw):
    share = raw.groupby(['Year', 'GP'])['Compound'].apply(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
    wet = set(share[share > 0.10].index)
    d = R.filter_laps(raw[[(y, g) not in wet for y, g in zip(raw.Year, raw.GP)]]).copy()
    d['DRS'] = ((d['Interval_front'] < 1.0) & (d['LapNumber'] > 2)).astype(int)
    d['Interval_front'] = np.exp(-d['Interval_front']); d['Interval_behind'] = np.exp(-d['Interval_behind'])
    d['LapNumber_2'] = d['LapNumber'] ** 2
    d['FirstLap_pos'] = d['Position'] * (d['LapNumber'] == 1.0)
    med = d.groupby(['Year', 'GP'])['LapTime'].transform('median')
    d = d[~((d['LapTime'] >= med + 15) & ~((d['PitIn'] == 1) | (d['PitOut'] == 1) | (d['LapNumber'] == 1)))]
    return d, wet


def main():
    raw = pd.concat(R.load_data(), ignore_index=True)
    d, wet = prep(raw)
    print('ediciones mojadas excluidas:', sorted(wet))
    rows = []
    for (y, g), dg in d.groupby(['Year', 'GP']):
        if 'VER' not in set(dg.Driver):
            print('sin VER:', y, g); continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m, dd = R.filtrar_outliers_iqr(dg, g)
        except Exception as e:
            print('falla', y, g, e); continue
        clean = dg[(dg.TrackStatus == 1.0) & (dg.PitIn == 0) & (dg.PitOut == 0)]
        lap_ref = float(clean.LapTime.median())
        n = dd.Driver.value_counts()
        pre = "C(Driver, Treatment(reference='VER'))[T."
        rows.append(dict(Year=y, GP=g, Driver='VER', coef=0.0, se=0.0, n_laps=int(n.get('VER', 0)), lap_ref=lap_ref))
        for k in m.params.index:
            if k.startswith(pre):
                drv = k[len(pre):-1]
                rows.append(dict(Year=y, GP=g, Driver=drv, coef=float(m.params[k]), se=float(m.bse[k]),
                                 n_laps=int(n.get(drv, 0)), lap_ref=lap_ref))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'race_coefs.csv'), index=False)
    print(df.groupby('Year').agg(carreras=('GP', 'nunique'), filas=('Driver', 'size')))
    print(df.coef.describe())


if __name__ == '__main__':
    main()
