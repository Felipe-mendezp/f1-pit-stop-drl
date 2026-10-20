"""Resumen: Figs. 6/7 v4 opcion3 (mundo = creencia) vs mundo opcion3 2021-24. Pareado por carrera."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, '..', '..', 'trained_models', 'rl_agents_v4_opcion3_20261007', 'selected')
BASE = {6: 'qmip_vs_drl_stochastic_results.csv', 7: 'all_gps_2024_qmip_results.csv'}
ALG = ['PPO', 'RecurrentPPO', 'QMIP']

def wide(d):
    d = d[d.algo.isin(ALG)].copy(); d['i'] = d.groupby(['display_name', 'algo']).cumcount()
    return d.pivot_table(index=['display_name', 'i'], columns='algo', values='final_position')

rows = []
for f in (6, 7):
    files = glob.glob(os.path.join(HERE, f'fig{f}_2024', '*.csv'))
    if not files: continue
    w = wide(pd.concat(pd.read_csv(x) for x in files))
    b = wide(pd.read_csv(os.path.join(S, BASE[f]))).loc[w.index]
    for gp, idx in list(w.groupby(level=0).groups.items()) + [('TOTAL', w.index)]:
        r = dict(fig=f, GP=gp, n=len(idx))
        for lab, x in (('v4', b.loc[idx]), ('w24', w.loc[idx])):
            for a in ALG: r[f'{lab}_{a}'] = x[a].mean()
            for a in ALG[:2]:
                dd = x.QMIP - x[a]; r[f'{lab}_gap_{a}'] = dd.mean(); r[f'{lab}_ci_{a}'] = 1.96 * dd.std() / np.sqrt(len(dd))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'summary_20261010.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['fig', 'GP'] + [f'{l}_{a}' for l in ('v4', 'w24') for a in ALG] + \
       [f'{l}_gap_{a}' for l in ('v4', 'w24') for a in ALG[:2]] + ['w24_ci_PPO', 'w24_ci_RecurrentPPO']
print(R[cols].round(2).to_string(index=False))
