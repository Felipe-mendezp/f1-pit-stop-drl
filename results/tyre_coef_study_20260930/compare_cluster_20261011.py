"""2026-10-11 — Opcion 3 OLS (pooled_20261004) vs opcion 3 con covarianza agrupada por stint
(pooled_cluster_20261011): coeficientes de los 8 GP del paper y plan optimo deterministico de un piloto solo
(enumeracion de 1 y 2 paradas, tiempo de vuelta = offset + pendiente x edad, perdida de parada time_stop de la
QMIP, regla de 2 compuestos; compuestos del filtro 5 % del mono-piloto). Escribe compare_cluster_20261011.csv."""
import os, sys, json, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
LAPS = {'Bahrain': 57, 'Saudi Arabian': 50, 'Miami': 57, 'Hungarian': 70, 'Belgian': 44, 'Dutch': 72,
        'Singapore': 62, 'United States': 56}
TSTOP = {'Bahrain': None}
filt = json.load(open(os.path.join(ROOT, 'data', 'compound_filter_one_driver_5pct_wetA_20261011.json')))['gps']
qp = json.load(open(os.path.join(HERE, 'time_stop_20261011.json')))


def stint(o, s, L):  # suma de offset + pendiente*edad (edad 1..L)
    return L * o + s * L * (L + 1) / 2


def best(co, comps, N, ts):
    out = []
    for k in (1, 2):
        for seq in itertools.product(comps, repeat=k + 1):
            if len(set(seq)) < 2: continue
            for cuts in itertools.combinations(range(1, N), k):
                Ls = np.diff([0, *cuts, N])
                t = sum(stint(*co[c], L) for c, L in zip(seq, Ls)) + k * ts
                out.append((t, '-'.join(f'{c}{L}' for c, L in zip(seq, Ls))))
    return min(out)


rows = []
for lab, f in [('OLS', 'pooled_20261004/coefs_opcion3.csv'), ('cluster', 'pooled_cluster_20261011/coefs_opcion3.csv')]:
    d = pd.read_csv(os.path.join(HERE, f))
    for g, N in LAPS.items():
        x = d[d.GP == g + ' Grand Prix'].set_index('compound')
        co = {c: (x.offset_vs_ref[c], x.slope[c]) for c in x.index}
        keep = filt[g + ' Grand Prix']['keep']
        t1, plan1 = best(co, keep, N, qp[g])
        rows.append(dict(GP=g, cov=lab, **{f'off_{c}': round(co[c][0], 3) for c in x.index},
                         **{f'sl_{c}': round(co[c][1], 4) for c in x.index}, plan_mono=plan1))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'compare_cluster_20261011.csv'), index=False)
pd.set_option('display.width', 250)
for g, x in R.groupby('GP', sort=False):
    print(x.dropna(axis=1, how='all').to_string(index=False)); print()
