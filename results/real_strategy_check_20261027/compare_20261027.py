"""2026-10-27 — Compara la posicion final simulada con la estrategia real (validate_real_strategy_20261027.py) con la
clasificacion real 2024 en pista (ultima vuelta de data/session_2024_V2.csv; abandonos ordenados por vueltas completadas).
Por escenario (fig6, fig7) y GP:
  - piloto del agente: posicion real, media e intervalo central del 90 % simulados, PIT aleatorizado;
  - todos los pilotos que terminaron en 2024 y casi siempre terminan en la simulacion: cobertura del intervalo del 90 %,
    PIT, error absoluto de la mediana, correlacion de Spearman entre la media simulada y la posicion real.
Escribe compare_agent_20261027.csv, compare_all_20261027.csv y compare_summary_20261027.csv en esta carpeta.
    ~/.venvs/f1-drl-v2/bin/python results/real_strategy_check_20261027/compare_20261027.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GPS = {'Bahrain_ALO': 'Bahrain Grand Prix', 'Saudi_Arabian_NOR': 'Saudi Arabian Grand Prix', 'Miami_OCO': 'Miami Grand Prix',
       'Hungarian_TSU': 'Hungarian Grand Prix', 'Belgian_GAS': 'Belgian Grand Prix', 'Dutch_STR': 'Dutch Grand Prix',
       'Singapore_HUL': 'Singapore Grand Prix', 'United_States_MAG': 'United States Grand Prix'}
ALIAS = {'BEA': 'SAI', 'COL': 'SAR', 'LAW': 'RIC'}   # reemplazos 2024 (el simulador usa el auto/nombre del titular)
rng = np.random.default_rng(0)


def actual(gp, s):
    g = s[s.GP == gp].sort_values('LapNumber')
    last = g.groupby('Driver').tail(1).copy()
    nl = g.LapNumber.max()
    last['finished'] = last.LapNumber >= nl - 3   # doblados terminan con 1-3 vueltas menos
    last = last.sort_values(['finished', 'LapNumber', 'Position'], ascending=[False, False, True])
    last['actual'] = range(1, len(last) + 1)
    last['driver'] = last.Driver.replace(ALIAS)
    return last.set_index('driver')[['actual', 'finished']]


def pit(x, a):
    """PIT aleatorizado para una distribucion discreta: P(X < a) + U * P(X = a)."""
    return (x < a).mean() + rng.uniform() * (x == a).mean()


def main():
    s = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'), usecols=['GP', 'Driver', 'LapNumber', 'Position'])
    rows = []
    for scen in ('fig7', 'fig6'):
        for key, gp in GPS.items():
            d = pd.read_csv(os.path.join(HERE, scen, f'{key}.csv'))
            act = actual(gp, s)
            miss = set(d.driver) ^ set(act.index)
            assert not miss, (gp, miss)
            for drv, x in d.groupby('driver'):
                pos = x.position.to_numpy()
                a, fin = act.loc[drv, 'actual'], act.loc[drv, 'finished']
                q05, q50, q95 = np.quantile(pos, [0.05, 0.5, 0.95], method='inverted_cdf')
                rows.append(dict(scen=scen, gp=key, driver=drv, agent=bool(x.agent.iloc[0]), actual=a, finished=fin,
                                 sim_dnf=x.is_dnf.mean(), mean=pos.mean(), q05=q05, q50=q50, q95=q95,
                                 inside90=q05 <= a <= q95, pit=pit(pos, a)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, 'compare_all_20261027.csv'), index=False)
    A = R[R.agent]
    A.to_csv(os.path.join(HERE, 'compare_agent_20261027.csv'), index=False)
    print('PILOTO DEL AGENTE')
    print(A[['scen', 'gp', 'driver', 'actual', 'mean', 'q05', 'q50', 'q95', 'inside90', 'pit']].round(2).to_string(index=False))
    # todos los pilotos que terminaron en 2024 y terminan en >= 95 % de las simulaciones
    C = R[R.finished & (R.sim_dnf <= 0.05)]
    summ = []
    for scen, g in C.groupby('scen', sort=False):
        rho = np.mean([spearmanr(h['mean'], h.actual).statistic for _, h in g.groupby('gp')])
        hist = np.histogram(g.pit, bins=5, range=(0, 1))[0]
        summ.append(dict(scen=scen, n=len(g), cover90=g.inside90.mean(), mae_median=(g.q50 - g.actual).abs().mean(),
                         bias_mean=(g['mean'] - g.actual).mean(), spearman_mean_gp=rho,
                         pit_quintiles=' '.join(map(str, hist)),
                         agent_cover90=A[A.scen == scen].inside90.mean()))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(HERE, 'compare_summary_20261027.csv'), index=False)
    print('\nTODOS LOS PILOTOS QUE TERMINARON (2024 y >= 95 % de las simulaciones)')
    print(S.round(3).to_string(index=False))
    out = C[~C.inside90].sort_values(['scen', 'gp'])
    print('\nFUERA DEL INTERVALO DEL 90 %')
    print(out[['scen', 'gp', 'driver', 'actual', 'mean', 'q05', 'q95']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
