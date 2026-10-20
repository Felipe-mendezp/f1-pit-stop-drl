"""2026-10-15 — Comparacion v6 frente a v7 de las Figs. 6 (abandonos reales) y 7 (mundo 2024).

Lee los resumenes que escribe results/world2024_v6_20261014/summarize_v6_20261014.py en cada carpeta
(summary_v6_20261014.csv: GP, src, algo, pos, pits, tard) y muestra, por figura, la posicion media, las paradas y el %
de carreras con parada en las ultimas 6 vueltas por GP y en promedio de los 8 GP, v6 | v7 | diferencia.
Salida: compare_v6_v7_20261015.csv en esta carpeta.
    python results/world2024_v7_20261015/compare_v6_v7_20261015.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = {'v6': os.path.join(ROOT, 'results', 'world2024_v6_20261014', 'summary_v6_20261014.csv'),
       'v7': os.path.join(HERE, 'summary_v6_20261014.csv')}
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
d = pd.concat([pd.read_csv(f).assign(version=v) for v, f in SRC.items()], ignore_index=True)
d = d[d.src.isin(['fig6_realdnf', 'fig7'])]
pd.set_option('display.width', 220)
rows = []
for fig, g in d.groupby('src'):
    for metric in ['pos', 'pits', 'tard']:
        t = g.pivot_table(index=['GP'], columns=['algo', 'version'], values=metric)
        t.loc['MEDIA 8 GP'] = t.mean()
        print(f'\n=== {fig}: {metric}')
        for a in ALGOS:
            if (a, 'v6') in t and (a, 'v7') in t:
                t[(a, 'dif')] = t[(a, 'v7')] - t[(a, 'v6')]
        cols = [(a, v) for a in ALGOS for v in ('v6', 'v7', 'dif') if (a, v) in t]
        print(t[cols].round(2).to_string())
        x = t[cols].copy(); x.columns = [f'{a}_{v}' for a, v in cols]
        rows.append(x.assign(fig=fig, metric=metric).reset_index())
pd.concat(rows).to_csv(os.path.join(HERE, 'compare_v6_v7_20261015.csv'), index=False)
