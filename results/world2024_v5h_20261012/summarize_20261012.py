"""2026-10-12 — Resumen Figs. 6/7 v5h (creencia opcion3h 2021-23, mundo opcion3h 2021-24), 5 algoritmos + QMIP.
Brecha = QMIP - algoritmo en posicion final, pareada por carrera (IC 95%). Referencia: v5 en mundo opcion3 2021-24
(results/world2024_v5_20261011, solo PPO/RPPO/QMIP). Escribe summary_20261012.csv."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.join(HERE, '..', 'world2024_v5_20261011')
ALG = ['PPO', 'RecurrentPPO', 'DQN', 'A2C', 'TRPO', 'QMIP']

def wide(folder, alg):
    d = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(folder, '*.csv')))], ignore_index=True)
    d = d[d.algo.isin(alg)].copy(); d['i'] = d.groupby(['display_name', 'algo']).cumcount()
    return d.pivot_table(index=['display_name', 'i'], columns='algo', values='final_position')

rows = []
for f in (6, 7):
    w = wide(os.path.join(HERE, f'fig{f}_2024'), ALG)
    b = wide(os.path.join(V5, f'fig{f}_2024'), ['PPO', 'RecurrentPPO', 'QMIP'])
    for gp, idx in list(w.groupby(level=0).groups.items()) + [('TOTAL', w.index)]:
        x = w.loc[idx]; r = dict(fig=f, GP=gp.replace(' GP', ''), n=len(idx))
        for a in ALG: r[a] = x[a].mean()
        for a in ALG[:-1]:
            dd = x.QMIP - x[a]; r[f'gap_{a}'] = dd.mean(); r[f'ci_{a}'] = 1.96 * dd.std() / np.sqrt(len(dd))
        y = b if gp == 'TOTAL' else b.loc[gp]
        for a in ('PPO', 'RecurrentPPO'): r[f'v5op3_gap_{a}'] = (y.QMIP - y[a]).mean()
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'summary_20261012.csv'), index=False)
pd.set_option('display.width', 250)
print(R[['fig', 'GP', 'n'] + ALG].round(2).to_string(index=False)); print()
g = ['fig', 'GP'] + [c for a in ALG[:-1] for c in (f'gap_{a}', f'ci_{a}')] + ['v5op3_gap_PPO', 'v5op3_gap_RecurrentPPO']
print(R[g].round(2).to_string(index=False))
