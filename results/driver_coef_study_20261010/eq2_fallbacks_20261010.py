"""2026-10-10 — Ec. 2 porcentual con respaldos, para el simulador 2024 (etapa B).

Correr desde la raiz del repo: python results/driver_coef_study_20261010/eq2_fallbacks_20261010.py
Entradas: eq2_features.csv (eq2_variants_20261010.py) y data/session_bestlaps_2021_2024_20261010.csv.

Especificacion completa (variante porcentual): z/pole ~ P1_pct + P2_pct + P3_pct + FP_pct + Q_pct.
Respaldos (se usa la especificacion mas rica cuyas variables existen para el piloto):
  - sin Q_pct si la clasificacion falta o fue con lluvia (Belgian 2024);
  - con las k < 3 carreras previas disponibles (novatos), y k = 0 solo con FP/Q.
Cada modelo se ajusta con ventana expansiva: solo carreras anteriores al GP objetivo (cruzando anios).

predict(year, gp, driver) -> dict con la prediccion (s/vuelta vs VER) y la especificacion usada.
main(): MSE de cada respaldo sobre 2023-24 (las mismas filas, forzando la falta de la variable).
"""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MIN_LAPS, MIN_TRAIN = 15, 30
PREV = ['P1_pct', 'P2_pct', 'P3_pct']


def spec(k, fp=True, q=True):
    return PREV[:k] + (['FP_pct'] if fp else []) + (['Q_pct'] if q else [])


class Eq2:
    def __init__(self):
        self.d = pd.read_csv(os.path.join(HERE, 'eq2_features.csv'))
        self.sample = self.d[(self.d.Driver != 'VER') & (self.d.n_laps >= MIN_LAPS)]
        bl = pd.read_csv(os.path.join(ROOT, 'data', 'session_bestlaps_2021_2024_20261010.csv'))
        bl.loc[bl.wet_share > 0.10, 'BestLapTime'] = np.nan
        fp = bl[bl.Session.str.startswith('Practice')].groupby(['Year', 'GP', 'Driver']).BestLapTime.min().rename('FP')
        q = bl[bl.Session == 'Qualifying'].groupby(['Year', 'GP', 'Driver']).BestLapTime.min().rename('Q')
        self.s = pd.concat([fp, q], axis=1).sort_index()
        self.rnd = bl.groupby(['Year', 'GP']).Round.first()
        self._cache = {}

    def features(self, year, gp, driver):
        """Variables de la Ec. 2 para (anio, GP, piloto), corra o no la carrera (p. ej. abandono en vuelta 1)."""
        t = year * 100 + int(self.rnd[(year, gp)])
        s = self.s.loc[(year, gp)]
        x = {}
        for c in ('FP', 'Q'):
            ver = s[c].get('VER', np.nan)
            x[c + '_pct'] = (s[c].get(driver, np.nan) - ver) / ver
        pole = s.Q.min()
        x['pole'] = pole if np.isfinite(pole) else s.FP.min()
        h = self.sample[(self.sample.Driver == driver) & (self.sample.t < t)].sort_values('t')
        for k in (1, 2, 3):
            x[f'P{k}_pct'] = h.z_pct.iloc[-k] if len(h) >= k else np.nan
        return t, x

    def model(self, t, cols):
        key = (t, tuple(cols))
        if key not in self._cache:
            tr = self.sample[self.sample.t < t].dropna(subset=cols + ['z_pct'])
            self._cache[key] = None if len(tr) < MIN_TRAIN else sm.OLS(tr.z_pct, sm.add_constant(tr[cols])).fit()
        return self._cache[key]

    def predict(self, year, gp, driver):
        if driver == 'VER':
            return dict(pred=0.0, spec='referencia', n_train=0)
        t, x = self.features(year, gp, driver)
        k = sum(np.isfinite(x[c]) for c in PREV)
        cols = spec(k, np.isfinite(x['FP_pct']), np.isfinite(x['Q_pct']))
        m = self.model(t, cols) if cols else None
        if m is None:
            return dict(pred=np.nan, spec='sin_modelo', n_train=0, **x)
        xv = pd.DataFrame([{c: x[c] for c in cols}])
        p = float(m.predict(sm.add_constant(xv, has_constant='add'))[0]) * x['pole']
        return dict(pred=p, spec='+'.join(cols), n_train=int(m.nobs), **x)


def main():
    E = Eq2()
    tg = E.sample[(E.sample.Year >= 2023)].dropna(subset=spec(3) + ['z_pct'])
    variants = {f'k{k}{"" if q else "_sinQ"}': spec(k, True, q) for k in (3, 2, 1, 0) for q in (True, False)}
    rows = []
    for (y, t), g in tg.groupby(['Year', 't']):
        for name, cols in variants.items():
            m = E.model(t, cols)
            if m is None: continue
            p = m.predict(sm.add_constant(g[cols], has_constant='add')) * g.pole
            rows += [dict(Year=y, variante=name, err2=e) for e in (p - g.coef) ** 2]
    R = pd.DataFrame(rows).groupby(['variante', 'Year']).err2.agg(['mean', 'size']).unstack('Year')
    R.to_csv(os.path.join(HERE, 'eq2_fallback_mse.csv'))
    print('MSE (s/vuelta)^2 de cada respaldo, mismas filas (todas con las 5 variables):')
    print(R.round(4).to_string())


if __name__ == '__main__':
    main()
