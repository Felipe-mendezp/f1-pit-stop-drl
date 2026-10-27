"""2026-10-27 — Analisis 3 (propuesta): en que fraccion de vueltas la recomendacion del look-ahead es "casi indiferente".
Lee los rollouts por vuelta del barrido E4 (../decision_support_v9r2smrealhist_20261026/<algo>/<gp>_<scen>_lap<n>.csv; PPO,
escenario 'real' = el de la Fig. 11: historia real hasta n-1, futuro simulado; M = 500, 4 acciones, rollouts pareados por
indice). Por vuelta: mejor accion por puntos medios, segunda mejor, diferencia pareada mejor - segunda +- 2 SE; tambien
por posicion final. Escribe ambiguity_laps_20261027.csv y ambiguity_summary_20261027.csv en esta carpeta.
    ~/.venvs/f1-drl-v2/bin/python results/decision_ambiguity_20261027/ambiguity_20261027.py [--algo PPO] [--scen real]
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'decision_support_v9r2smrealhist_20261026')


def lap_stats(df, metric, higher_better):
    w = df.pivot(index='rollout', columns='action', values=metric)
    m = w.mean()
    order = (m.sort_values(ascending=not higher_better)).index.tolist()
    b, s = order[0], order[1]
    d = (w[b] - w[s]) * (1 if higher_better else -1)
    return b, s, d.mean(), 2 * d.std(ddof=1) / np.sqrt(len(d)), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', default='PPO')
    ap.add_argument('--scen', default='real')
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, a.algo, f'*_{a.scen}_lap*.csv'))):
        gp, n = re.match(rf'(.+)_{a.scen}_lap(\d+)\.csv', os.path.basename(f)).groups()
        df = pd.read_csv(f)
        lab = df.drop_duplicates('action').set_index('action').action_label
        bp, sp, dp, sep, mp = lap_stats(df, 'points', True)
        bq, sq, dq, seq, _ = lap_stats(df, 'final_position', False)
        # decision de parar: no parar vs la mejor parada; eleccion de compuesto: mejor parada vs segunda parada
        w = df.pivot(index='rollout', columns='action', values='points')
        pits = w.drop(columns=0).mean().sort_values(ascending=False).index.tolist()
        dstop = w[pits[0]] - w[0]
        dcomp = w[pits[0]] - w[pits[1]]
        se = lambda x: 2 * x.std(ddof=1) / np.sqrt(len(x))
        rows.append(dict(gp=gp, lap=int(n), best_pit=lab[pits[0]], d_stop=dstop.mean(), se2_stop=se(dstop),
                         stop_better=dstop.mean() > se(dstop), stop_tie=abs(dstop.mean()) <= se(dstop),
                         d_comp=dcomp.mean(), se2_comp=se(dcomp), comp_tie=dcomp.mean() <= se(dcomp),
                         best_pts=lab[bp], second_pts=lab[sp], d_pts=dp, se2_pts=sep,
                         tie_pts=dp <= sep, all_zero_pts=bool((mp == 0).all()),
                         best_pos=lab[bq], second_pos=lab[sq], d_pos=dq, se2_pos=seq, tie_pos=dq <= seq))
    L = pd.DataFrame(rows).sort_values(['gp', 'lap'])
    L.to_csv(os.path.join(HERE, 'ambiguity_laps_20261027.csv'), index=False)
    L['window'] = L.stop_better | L.stop_tie   # vueltas en que parar es mejor o indistinguible de no parar
    W = L[L.window]
    S = L.groupby('gp').agg(laps=('lap', 'size'), tie_any=('tie_pts', 'mean'), stop_better=('stop_better', 'sum'),
                             stop_tie=('stop_tie', 'sum'), all_zero=('all_zero_pts', 'mean'))
    S['comp_tie_in_window'] = W.groupby('gp').comp_tie.sum()
    S.loc['ALL'] = [len(L), L.tie_pts.mean(), L.stop_better.sum(), L.stop_tie.sum(), L.all_zero_pts.mean(),
                    W.comp_tie.sum()]
    S.to_csv(os.path.join(HERE, 'ambiguity_summary_20261027.csv'))
    print(S.round(3).to_string())
    print(f'\nventana de parada (parar mejor o indistinguible de no parar): {len(W)} vueltas; '
          f'parar mejor: {int(L.stop_better.sum())}; indistinguible: {int(L.stop_tie.sum())}; '
          f'compuesto indistinguible dentro de la ventana: {int(W.comp_tie.sum())}')
    print(W[['gp', 'lap', 'best_pit', 'd_stop', 'se2_stop', 'stop_better', 'd_comp', 'se2_comp', 'comp_tie']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
