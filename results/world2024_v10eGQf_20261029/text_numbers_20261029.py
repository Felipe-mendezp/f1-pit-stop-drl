"""2026-10-29 — Cifras del texto de las Figs. 6-7 (v10eGQf): promedio por metodo (todos los GP y sin Miami), diferencia
pareada con la QMIP y, en la Fig. 7, con la estrategia real repetida (results/real_strategy_check_v10eGQf_20261029/fig7,
mismas semillas). IC del 95 % = 1,96 x error estandar de la diferencia pareada carrera a carrera.
    ~/.venvs/f1-drl-v2/bin/python results/world2024_v10eGQf_20261029/text_numbers_20261029.py"""
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VAL = os.path.join(ROOT, 'results', 'real_strategy_check_v10eGQf_20261029', 'fig7')
GPS = ['Bahrain_ALO', 'Saudi_Arabian_NOR', 'Miami_OCO', 'Hungarian_TSU', 'Belgian_GAS', 'Dutch_STR', 'Singapore_HUL',
       'United_States_MAG']
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']


def load(fig):
    rows = []
    for g in GPS:
        d = pd.read_csv(os.path.join(HERE, fig, g + '.csv'))
        for a, x in d.groupby('algo'):
            rows.append(pd.DataFrame({'gp': g, 'algo': a, 'sim': np.arange(len(x)), 'pos': x.final_position.values}))
        if fig == 'fig7_2024':
            v = pd.read_csv(os.path.join(VAL, g + '.csv'))
            v = v[v.agent].sort_values('sim')
            rows.append(pd.DataFrame({'gp': g, 'algo': 'Actual', 'sim': np.arange(len(v)), 'pos': v.position.values}))
    return pd.concat(rows).pivot_table(index=['gp', 'sim'], columns='algo', values='pos')


def diff(P, ref, gps):
    P = P.loc[gps]
    out = {}
    for a in ALGOS + ['Actual']:
        if a not in P or a == ref:
            continue
        # promedio sobre GP de la diferencia media; error estandar con varianza por GP (como ci_table)
        m = np.mean([(P.loc[g][a] - P.loc[g][ref]).mean() for g in gps])
        se = np.sqrt(sum((P.loc[g][a] - P.loc[g][ref]).var(ddof=1) / len(P.loc[g]) for g in gps)) / len(gps)
        out[a] = f'{m:+.2f} ± {1.96 * se:.2f}'
    return out


for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    P = load(fig)
    for lab, gps in (('8 GP', GPS), ('sin Miami', [g for g in GPS if g != 'Miami_OCO'])):
        print(f'== {fig} [{lab}] media:', P.loc[gps].mean().round(2).to_dict())
        print('   dif. con QMIP:', diff(P, 'QMIP', gps))
        if fig == 'fig7_2024':
            print('   dif. con Actual:', diff(P, 'Actual', gps))
    if fig == 'fig7_2024':
        m = P.groupby('gp').mean()
        print('   GP donde Actual es mejor que todos los metodos:',
              [g for g in GPS if all(m.loc[g, 'Actual'] < m.loc[g, a] for a in ALGOS)])
        print(m.round(2))
