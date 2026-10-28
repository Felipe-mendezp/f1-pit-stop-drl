"""2026-10-28 — Peso de las vueltas de empuje de clasificacion (pedido del autor: quizas x2).
Edad con que llega un juego usado: A0_psi = push_o + psi x push_q + RHO_psi x (vueltas no encontradas), con push_o = vueltas
de empuje de practicas y sprint, push_q = de clasificacion (Q, SQ, Shootout) y RHO_psi = (sum push_o + psi sum push_q) /
vueltas encontradas. TyreLife = A0_psi + vueltas de la tanda en carrera (sin SC/VSC: theta = 0; o theta = 0,25).
psi = 1 es GQ; se compara contra FastF1 y contra G (edad previa de FastF1, sin SC/VSC).
Mismo procedimiento que compare_20261028.py.
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/tyrelife_effective_20261028/grid_quali_20261028.py
Salida: grid_quali.csv, grid_quali_summary.txt
"""
import os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('cmp', os.path.join(HERE, 'compare_20261028.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import numpy as np, pandas as pd
P, KEY = M.P, M.KEY
PSI = [0, 0.5, 1, 1.5, 2, 3, 4]
THETA = [0, 0.25]


def main():
    A = pd.read_csv(os.path.join(HERE, 'ages_2021_2024.csv.gz'))
    T = pd.read_csv(os.path.join(HERE, 'stint_starts.csv')); U = T[T.a0 > 0]
    matched = (U.a0 - U.unmatched).sum()
    sc = A.TL_Q - A.TL_GQ
    race = A.TL_GQ - A.a0_ef                                   # vueltas de la tanda en carrera sin SC/VSC
    names = []
    for th in THETA:
        n = f'G_{th}'; A[n] = A.TL_G + th * sc; names.append((n, th, 'G'))
        for psi in PSI:
            rho = ((U.push - U.push_q).sum() + psi * U.push_q.sum()) / matched
            a0 = np.where(A.a0 > 0, A.push - A.push_q + psi * A.push_q + rho * A.unmatched, 0.0)
            n = f'Q_{th}_{psi}'; A[n] = a0 + race + th * sc; names.append((n, th, psi))
    M.VARS = [n for n, _, _ in names]
    rows = []
    for d, lab in ((P.hist_laps_A(), '2021-23'), (P.laps_2024_A(), '2024')):
        F, _ = M.run(d, lab, A[KEY + ['TL_raw'] + M.VARS])
        for n, th, psi in names:
            for sub, y in (('todos', F), ('8 GP paper', F[F.GP.isin(P.PAPER)])):
                rows.append(dict(set=lab, sub=sub, theta=th, psi=psi,
                                 sse_rel=y[f'sse_{n}'].sum() / y.sse_raw.sum() - 1))
    G = pd.DataFrame(rows); G.to_csv(os.path.join(HERE, 'grid_quali.csv'), index=False)
    out = [f'RHO(psi=1) = {((U.push).sum()) / matched:.3f}; vueltas de empuje de clasificacion en juegos usados: '
           f'{U.push_q.sum():.0f} de {U.push.sum():.0f}']
    for (lab, sub), x in G.groupby(['set', 'sub']):
        t = (x.pivot(index='theta', columns='psi', values='sse_rel') * 100).round(2)
        out.append(f'\n{lab} {sub}: cambio de SSE (%) frente a FastF1; filas theta (peso SC/VSC), columnas psi (peso de '
                   f'vuelta de empuje de clasificacion; G = edad previa de FastF1)\n' + t.to_string())
    print('\n'.join(out)); open(os.path.join(HERE, 'grid_quali_summary.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
