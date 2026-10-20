"""2026-10-11 — Resumen Figs. 6/7 v5 (simulador con coef. de piloto Ec. 2) vs v4 opcion3. Pareado por carrera.

Correr desde la raiz del repo: python results/figs_v5_20261010/summarize_20261010.py
Escribe summary_20261010.csv (posiciones medias y brecha QMIP - DRL con IC 95%) y stops_20261010.csv.
"""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, '..', '..', 'trained_models', 'rl_agents_v4_opcion3_20261007', 'selected')
BASE = {6: 'qmip_vs_drl_stochastic_results.csv', 7: 'all_gps_2024_qmip_results.csv'}
ALG = ['PPO', 'RecurrentPPO', 'QMIP']


def wide(d, col='final_position'):
    d = d[d.algo.isin(ALG)].copy(); d['i'] = d.groupby(['display_name', 'algo']).cumcount()
    return d.pivot_table(index=['display_name', 'i'], columns='algo', values=col)


rows, stops = [], []
for f in (6, 7):
    v5 = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(HERE, f'fig{f}_none', '*.csv')))], ignore_index=True)
    w = wide(v5)
    b = wide(pd.read_csv(os.path.join(S, BASE[f]))).loc[w.index]
    for gp, idx in list(w.groupby(level=0).groups.items()) + [('TOTAL', w.index)]:
        r = dict(fig=f, GP=gp, n=len(idx))
        for lab, x in (('v4', b.loc[idx]), ('v5', w.loc[idx])):
            for a in ALG: r[f'{lab}_{a}'] = x[a].mean()
            for a in ALG[:2]:
                dd = x.QMIP - x[a]; r[f'{lab}_gap_{a}'] = dd.mean(); r[f'{lab}_ci_{a}'] = 1.96 * dd.std() / np.sqrt(len(dd))
        rows.append(r)
    st = v5[v5.algo.isin(ALG)].pit_stops.clip(upper=4)
    s = pd.crosstab(v5.loc[st.index, 'algo'], st, normalize='index').mul(100).round(1)
    s.insert(0, 'fig', f); stops.append(s.reset_index())

R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'summary_20261010.csv'), index=False)
P = pd.concat(stops, ignore_index=True); P.to_csv(os.path.join(HERE, 'stops_20261010.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['fig', 'GP', 'n'] + [f'{l}_{a}' for l in ('v4', 'v5') for a in ALG] + \
       [f'{l}_{k}_{a}' for l in ('v4', 'v5') for a in ALG[:2] for k in ('gap', 'ci')]
print(R[cols].round(2).to_string(index=False))
print('\n% de carreras por numero de CAMBIOS DE COMPUESTO (pit_stops de los evaluadores; no cuenta paradas S->S), v5:')
print(P.to_string(index=False))
