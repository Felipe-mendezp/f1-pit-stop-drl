"""2026-10-28 — MSE de los respaldos de la Ec. 2 (texto de app:adjust_driver: "Dropping X_Q changes the error by at most
0.02 ... up to about 0.36 when none is available"), como main() de eq2_fallbacks_20261010.py pero con la clase de
eq2_fallbacks_20261014.py (regla de VER >= 15 vueltas). control/ debe reproducir
../driver_coef_study_20261010/eq2_fallback_mse_20261014.csv. Salida: <version>/eq2_fallback_mse_20261014.csv
"""
import os, importlib.util
import pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
for v in ('control', 'v10', 'v10eGQ'):   # 28-oct noche: + v10eGQ
    s = importlib.util.spec_from_file_location('eq2_' + v, os.path.join(HERE, v, 'eq2_fallbacks_20261014.py'))
    M = importlib.util.module_from_spec(s); s.loader.exec_module(M)
    E = M.Eq2(); spec = M.spec
    tg = E.sample[(E.sample.Year >= 2023)].dropna(subset=spec(3) + ['z_pct'])
    variants = {f'k{k}{"" if q else "_sinQ"}': spec(k, True, q) for k in (3, 2, 1, 0) for q in (True, False)}
    rows = []
    for (y, t), g in tg.groupby(['Year', 't']):
        for name, cols in variants.items():
            m = E.model(t, cols)
            if m is None: continue
            p = m.predict(sm.add_constant(g[cols], has_constant='add')) * g.pole
            rows += [dict(Year=y, v=name, err2=e) for e in (p - g.coef) ** 2]
    R = pd.DataFrame(rows).groupby(['v', 'Year']).err2.mean().unstack('Year')
    R.to_csv(os.path.join(HERE, v, 'eq2_fallback_mse_20261014.csv'))
    print(v); print(R.round(4).to_string())
