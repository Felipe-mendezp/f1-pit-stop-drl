"""2026-10-30 — Agentes actuales (retrained_v10eGQf_20261029/selected) en el mundo vigente frente al mundo con efecto
piloto-carrera sigma 0.24 (world2024_r24): posicion media y de por GP y metodo; diferencia con la QMIP pareada carrera a
carrera (+- IC 95 %), 8 GP y sin Miami.
    python results/dispersion_hyp_20261030/compare_agents_20261030.py
"""
import os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
W = {'vigente': os.path.join(ROOT, 'results', 'world2024_v10eGQf_20261029'), 'r24': os.path.join(HERE, 'world2024_r24')}
ALG = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    rows, diffs = [], []
    for w, d in W.items():
        for f in sorted(glob.glob(os.path.join(d, fig, '*.csv'))):
            gp = os.path.basename(f)[:-4]; x = pd.read_csv(f)
            x['i'] = x.groupby('algo').cumcount()
            p = x.pivot(index='i', columns='algo', values='final_position')
            for a in ALG:
                if a in p:
                    rows.append(dict(world=w, gp=gp, algo=a, txt=f'{p[a].mean():.2f}±{p[a].std():.2f}'))
                    if a != 'QMIP':
                        diffs.append(pd.DataFrame(dict(world=w, gp=gp, algo=a, d=p[a] - p['QMIP'])))
    R = pd.DataFrame(rows); D = pd.concat(diffs)
    print(f'\n===== {fig}: posicion media ± de =====')
    print(R.pivot_table(index=['gp', 'world'], columns='algo', values='txt', aggfunc='first')[ALG].to_string())
    out = []
    for (w, a), g in D.groupby(['world', 'algo']):
        for lab, h in (('8 GP', g), ('sin Miami', g[g.gp != 'Miami_OCO'])):
            out.append(dict(world=w, algo=a, set=lab, txt=f'{h.d.mean():+.2f} ± {1.96 * h.d.std() / np.sqrt(len(h)):.2f}'))
    print(f'\n----- {fig}: diferencia con la QMIP (negativo = mejor) -----')
    print(pd.DataFrame(out).pivot_table(index=['set', 'world'], columns='algo', values='txt', aggfunc='first')[ALG[:-1]].to_string())
