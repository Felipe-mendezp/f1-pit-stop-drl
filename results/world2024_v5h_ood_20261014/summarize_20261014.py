"""Resumen de las intervenciones DNF (2026-10-14): base = primeras 200 carreras de
results/world2024_v5h_pits_20261014/fig7_2024 (mismas semillas). Por GP, modo y metodo: posicion media,
paradas medias reales y % de carreras con alguna parada en las ultimas 6 vueltas."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(os.path.dirname(HERE), 'world2024_v5h_pits_20261014')
N = {'Saudi_Arabian_NOR': 50, 'Miami_OCO': 57, 'Singapore_HUL': 62}
A = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
rows = []
for g, n in N.items():
    for m, p in [('base', os.path.join(BASE, 'fig7_2024', f'{g}.csv')), ('nodnf', os.path.join(HERE, 'nodnf', 'fig7_2024', f'{g}.csv')),
                 ('dnfobs', os.path.join(HERE, 'dnfobs', 'fig7_2024', f'{g}.csv'))]:
        if not os.path.isfile(p):
            continue
        d = pd.read_csv(p)
        for a in A:
            x = d[d.algo == a].head(200)
            laps = x.pit_laps.fillna('').astype(str).apply(lambda s: [int(float(v)) for v in s.split()])
            rows.append(dict(GP=g.split('_')[0], modo=m, algo=a, pos=x.final_position.mean(), pits=x.n_pits_all.mean(),
                             tard=100 * np.mean([any(l > n - 6 for l in ls) for ls in laps])))
d = pd.DataFrame(rows)
for c, t in [('pos', 'posicion media'), ('pits', 'paradas medias reales'), ('tard', '% carreras con parada en las ultimas 6 vueltas')]:
    print(f'\n=== {t}')
    print(d.pivot_table(index=['GP', 'modo'], columns='algo', values=c, sort=False)[A].round(2).to_string())
