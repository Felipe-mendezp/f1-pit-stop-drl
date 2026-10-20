"""2026-10-15 — Validacion fuera de muestra de los coeficientes de neumatico v6 frente a v7 contra la Ec. 1 de cada
GP estimada solo con la carrera 2024 (misma vara y metricas que pooled_20261004.py: pendientes y diferencias entre
compuestos a 0, 10 y 20 vueltas). Salida: results/sim_v7_20261015/validation_2024_v7_20261015.csv
"""
import os, sys, importlib.util
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
d24 = P.laps_2024_A()
Y = {}
for g in P.MAP:
    if g in set(d24.GP):
        try: Y[g] = P.fit_gp(d24, g)
        except Exception as e: print('2024 sin regresion:', g, e)
SRC = {'v6': os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_hier_20261011', 'coefs_opcion3h.csv'),
       'v7': os.path.join(HERE, 'tyre_v7', 'coefs_opcion3h.csv')}
GPS = sorted(set(pd.read_csv(SRC['v6']).GP))
T = P.truth_table(Y, GPS)
res = []
for k, f in SRC.items():
    C = pd.read_csv(f)
    o = {(r.GP, r.compound): r.offset_vs_ref for r in C.itertuples()}
    s = {(r.GP, r.compound): r.slope for r in C.itertuples()}
    res.append(T.assign(p=P.predict(T, o, s), version=k))
V = pd.concat(res); V['err'] = V.p - V.t
V.to_csv(os.path.join(HERE, 'validation_2024_v7_20261015.csv'), index=False)
rows = []
for lab, sub in [('todos', V), ('8 GP paper', V[V.GP.isin(P.PAPER)])]:
    for k, x in sub.groupby('version'):
        dl = x[x.kind == 'delta'].dropna(subset=['p']); sl = x[x.kind == 'slope'].dropna(subset=['p'])
        rows.append(dict(conjunto=lab, version=k, n_delta=len(dl), MAE_delta=dl.err.abs().mean(),
                         MAE_delta_pond=np.average(dl.err.abs(), weights=1 / dl.se ** 2),
                         n_slope=len(sl), MAE_slope=sl.err.abs().mean(), sesgo_slope=sl.err.mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))
