"""2026-10-11 — Resumen prueba world-2024 sobre v5: Figs. 6/7 v5 (mundo = creencia, results/figs_v5_20261010)
vs mundo con neumaticos opcion3 2021-24. Pareado por carrera. Escribe summary_20261011.csv."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'figs_v5_20261010')
ALG = ['PPO', 'RecurrentPPO', 'QMIP']

def wide(d):
    d = d[d.algo.isin(ALG)].copy(); d['i'] = d.groupby(['display_name', 'algo']).cumcount()
    return d.pivot_table(index=['display_name', 'i'], columns='algo', values='final_position')

def load(folder):
    return pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(folder, '*.csv')))], ignore_index=True)

rows = []
for f in (6, 7):
    if not glob.glob(os.path.join(HERE, f'fig{f}_2024', '*.csv')): continue
    w = wide(load(os.path.join(HERE, f'fig{f}_2024')))
    b = wide(load(os.path.join(BASE, f'fig{f}_none'))).loc[w.index]
    for gp, idx in list(w.groupby(level=0).groups.items()) + [('TOTAL', w.index)]:
        r = dict(fig=f, GP=gp, n=len(idx))
        for lab, x in (('v5', b.loc[idx]), ('w24', w.loc[idx])):
            for a in ALG: r[f'{lab}_{a}'] = x[a].mean()
            for a in ALG[:2]:
                dd = x.QMIP - x[a]; r[f'{lab}_gap_{a}'] = dd.mean(); r[f'{lab}_ci_{a}'] = 1.96 * dd.std() / np.sqrt(len(dd))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'summary_20261011.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['fig', 'GP', 'n'] + [f'{l}_{a}' for l in ('v5', 'w24') for a in ALG] + \
       [f'{l}_{k}_{a}' for l in ('v5', 'w24') for a in ALG[:2] for k in ('gap', 'ci')]
print(R[cols].round(2).to_string(index=False))
