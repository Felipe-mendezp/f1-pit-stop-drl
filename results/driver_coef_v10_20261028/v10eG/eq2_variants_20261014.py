"""2026-10-14 — Tabla del apendice de la Ec. 2 (4 especificaciones + 2 lineas base) con la regla de v6: se excluyen
las carreras en que el piloto de referencia (VER) tiene < 15 vueltas limpias (Australia 2024), como observacion y
como carrera previa (P1-P3 = 3 carreras validas previas del piloto, recalculadas en todas las filas).
Mismo protocolo que eq2_variants_20261010.py (ventana expansiva, misma muestra para todos los metodos).
Entrada: eq2_features.csv. Salida: eq2_mse_summary_20261014.csv.
    python results/driver_coef_study_20261010/eq2_variants_20261014.py
"""
import os, importlib.util
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('v', os.path.join(HERE, 'eq2_variants_20261010.py'))
V = importlib.util.module_from_spec(_s)
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
_s.loader.exec_module(V)
_f = importlib.util.spec_from_file_location('f', os.path.join(HERE, 'eq2_fallbacks_20261014.py'))
F = importlib.util.module_from_spec(_f); _f.loader.exec_module(F)

d = pd.read_csv(os.path.join(HERE, 'eq2_features.csv'))
bad = F.bad_races()
d = d[[(y, g) not in bad for y, g in zip(d.Year, d.GP)]]
sample = d[(d.Driver != 'VER') & (d.n_laps >= V.MIN_LAPS)].sort_values(['Driver', 't']).copy()
for k in (1, 2, 3):
    sample[f'P{k}'] = sample.groupby('Driver').coef.shift(k)
    sample[f'P{k}_pct'] = sample.groupby('Driver').z_pct.shift(k)
preds = []
for (y, g, t), tg in sample[sample.Year >= 2023].groupby(['Year', 'GP', 't']):
    tr_all = sample[sample.t < t]
    r = tg[['Year', 'GP', 'Driver', 'coef', 'P1', 'hist']].copy()
    for name, cols in V.SPECS.items():
        ycol = 'z_pct' if name == 'porcentual' else 'coef'
        tr = tr_all.dropna(subset=cols + [ycol]); r[name] = np.nan
        if len(tr) < V.MIN_TRAIN: continue
        m = sm.OLS(tr[ycol], sm.add_constant(tr[cols])).fit()
        ok = tg[cols].notna().all(axis=1)
        if ok.any():
            p = m.predict(sm.add_constant(tg.loc[ok, cols], has_constant='add'))
            r.loc[ok, name] = p * (tg.loc[ok, 'pole'] if name == 'porcentual' else 1)
    preds.append(r)
P = pd.concat(preds).rename(columns={'P1': 'sin_correccion_prev1', 'hist': 'sin_correccion_hist'})
meth = list(V.SPECS) + ['sin_correccion_prev1', 'sin_correccion_hist']
C = P.dropna(subset=meth)
S = pd.DataFrame([dict(muestra=str(y), n=len(C[C.Year == y]),
                       **{m: ((C[C.Year == y][m] - C[C.Year == y].coef) ** 2).mean() for m in meth}) for y in (2023, 2024)])
S.to_csv(os.path.join(HERE, 'eq2_mse_summary_20261014.csv'), index=False)
print('excluidas:', sorted(bad)); print(S.round(3).to_string(index=False))
