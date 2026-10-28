"""2026-10-28 — Grilla de pesos para la edad efectiva: TyreLife = GQ + theta x (vueltas bajo SC/VSC en la tanda)
+ phi x (vueltas lentas previas del juego), es decir theta = phi = 1 es la edad de FastF1 y theta = phi = 0 es GQ.
Mismo procedimiento que compare_20261028.py (muestra fijada con raw por GP, 2021-23 filtro A y 2024).
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/tyrelife_effective_20261028/grid_20261028.py
Salida: grid.csv, grid_summary.txt
"""
import io, os, sys, contextlib, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('cmp', os.path.join(HERE, 'compare_20261028.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import numpy as np, pandas as pd
R, P, KEY = M.R, M.P, M.KEY
W = [0, 0.25, 0.5, 0.75, 1]


def main():
    A = pd.read_csv(os.path.join(HERE, 'ages_2021_2024.csv.gz'))
    sc, slow = A.TL_Q - A.TL_GQ, A.TL_raw - A.TL_Q
    names = []
    for th in W:
        for ph in W:
            n = f'TL_{th}_{ph}'; A[n] = A.TL_GQ + th * sc + ph * slow; names.append((n, th, ph))
    M.VARS = [n for n, _, _ in names]
    rows = []
    for d, lab in ((P.hist_laps_A(), '2021-23'), (P.laps_2024_A(), '2024')):
        F, _ = M.run(d, lab, A[KEY + ['TL_raw'] + M.VARS])
        for n, th, ph in names:
            for sub, y in (('todos', F), ('8 GP paper', F[F.GP.isin(P.PAPER)])):
                rows.append(dict(set=lab, sub=sub, theta=th, phi=ph, sse_rel=y[f'sse_{n}'].sum() / y.sse_raw.sum() - 1))
    G = pd.DataFrame(rows); G.to_csv(os.path.join(HERE, 'grid.csv'), index=False)
    out = []
    for (lab, sub), x in G.groupby(['set', 'sub']):
        t = x.pivot(index='theta', columns='phi', values='sse_rel') * 100
        b = x.loc[x.sse_rel.idxmin()]
        out.append(f'\n{lab} {sub}: cambio de SSE (%) frente a FastF1 (filas theta = peso de vuelta SC/VSC, columnas phi = '
                   f'peso de vuelta lenta previa); minimo en theta {b.theta}, phi {b.phi}\n' + t.round(2).to_string())
    print('\n'.join(out)); open(os.path.join(HERE, 'grid_summary.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
