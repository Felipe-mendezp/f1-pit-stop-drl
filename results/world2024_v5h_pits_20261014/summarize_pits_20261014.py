"""Resumen de paradas reales (n_pits_all, pit_laps) de las Figs. 6/7 v5h de 4 GP — 2026-10-14.
Verifica que las posiciones coincidan con results/world2024_v5h_20261012 y resume, por GP y metodo:
  % de carreras con alguna parada al mismo compuesto (n_pits_all > cambios de compuesto) y, entre parentesis,
  % con esa parada en las ultimas 6 vueltas; paradas medias reales vs las del CSV antiguo (pit_stops)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ORIG = os.path.join(os.path.dirname(HERE), 'world2024_v5h_20261012')
N = {'Bahrain_ALO': 57, 'Saudi_Arabian_NOR': 50, 'Miami_OCO': 57, 'Singapore_HUL': 62}
A = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
for f in (6, 7):
    rows = []
    for g, n in N.items():
        p = os.path.join(HERE, f'fig{f}_2024', f'{g}.csv'); new = pd.read_csv(p)
        if 'n_pits_all' not in new:
            print(f'(fig {f} {g}: sin columnas nuevas, se omite)'); continue
        old = pd.read_csv(os.path.join(ORIG, f'fig{f}_2024', f'{g}.csv'))
        assert (new.final_position.values == old.final_position.values).all() and (new.algo.values == old.algo.values).all(), (f, g)
        for a in A:
            x = new[new.algo == a]
            laps = x.pit_laps.fillna('').astype(str).apply(lambda s: [int(float(v)) for v in s.split()])
            last = x['last_lap_pit'] if 'last_lap_pit' in x else 0
            extra = (x.n_pits_all - x.pit_stops - last) > 0
            fin6 = [e and len(ls) >= 2 and n - ls[-1] < 6 for e, ls in zip(extra, laps)]
            rows.append(dict(GP=g.split('_')[0], algo=a, extra=100 * extra.mean(), fin6=100 * np.mean(fin6),
                             real=x.n_pits_all.mean(), csv=x.pit_stops.mean()))
    d = pd.DataFrame(rows)
    print(f'\n=== Fig {f}: % carreras con parada al mismo compuesto (en parentesis: con esa ultima parada en las ultimas 6 vueltas)')
    pv = lambda c: d.pivot(index='GP', columns='algo', values=c)[A]
    print((pv('extra').round(0).astype(int).astype(str) + ' (' + pv('fin6').round(0).astype(int).astype(str) + ')').to_string())
    print('paradas medias reales (CSV antiguo):')
    print((pv('real').round(2).astype(str) + ' (' + pv('csv').round(2).astype(str) + ')').to_string())
print('\nposiciones identicas a la corrida original: OK')
