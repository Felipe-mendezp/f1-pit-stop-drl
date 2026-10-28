"""2026-10-27 — Coeficientes de neumatico del simulador v10 = misma cadena que results/sim_v7_20261015/tyre_v7_20261015.py
(Ec. 1 por GP, offsets opcion3, pendientes opcion3h con tau por ML, y "mundo 2024" con 2021-24), sin las vueltas de paso
por el pit lane sin parada (common_v10_20261027.patch_regressions). Requiere F1_TYRELIFE_2021=v10 (v7 con CONTROL=1).
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/tyre_v10_20261027.py
Salida: results/sim_v10_20261027/tyre_v10/{coefs_opcion3h.csv, coefs_opcion3h_2021_24.csv, tau.txt} y comparacion con v7.
"""
import importlib.util, os, sys
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), HERE]
import config, compat; compat.setup_all()
import pandas as pd
import models.regressions as R
import common_v10_20261027 as C

assert os.environ.get('F1_TYRELIFE_2021') == ('v7' if os.environ.get('CONTROL') == '1' else 'v10'), 'F1_TYRELIFE_2021=v10 (v7 con CONTROL=1)'
C.patch_regressions(R)
spec = importlib.util.spec_from_file_location('t7', os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7_20261015.py'))
T7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T7)
OUT = os.path.join(HERE, 'tyre_v10'); os.makedirs(OUT, exist_ok=True)
V7 = os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7')


def main():
    P = T7.P
    assert P.R is R and R._v10_patched
    dh, d24 = P.hist_laps_A(), P.laps_2024_A()
    print('vueltas 2021-23 (filtro A, sin pasos):', len(dh), '| 2024:', len(d24))
    H = {g: P.fit_gp(dh, g) for g in P.MAP if g in set(dh.GP)}
    GPS = [g for g in P.MAP if g in H]
    Cf, tau = T7.coefs(dh, GPS, H)
    Cf.to_csv(os.path.join(OUT, 'coefs_opcion3h.csv'), index=False)
    d = pd.concat([dh, d24], ignore_index=True, sort=False)
    H4 = {g: P.fit_gp(d, g) for g in GPS}
    C4, tau4 = T7.coefs(d, GPS, H4)
    C4.to_csv(os.path.join(OUT, 'coefs_opcion3h_2021_24.csv'), index=False)
    with open(os.path.join(OUT, 'tau.txt'), 'w') as f:
        f.write(f'tau_2021_23 {tau:.6f}\ntau_2021_24 {tau4:.6f}\n')
    print(f'v10: tau 2021-23 = {tau:.4f} | 2021-24 = {tau4:.4f}')
    for new, name in [(Cf, 'coefs_opcion3h.csv'), (C4, 'coefs_opcion3h_2021_24.csv')]:
        old = pd.read_csv(os.path.join(V7, name))
        m = old.merge(new, on=['GP', 'ref', 'compound'], suffixes=('_v7', '_v10'))
        m['d_off'] = m.offset_vs_ref_v10 - m.offset_vs_ref_v7; m['d_slope'] = m.slope_v10 - m.slope_v7
        m.to_csv(os.path.join(OUT, name.replace('.csv', '_vs_v7.csv')), index=False)
        pg = m[m.GP.isin(T7.P.PAPER)] if hasattr(T7.P, 'PAPER') else m
        print(name, 'filas', len(m), '/', len(old), '| max |d offset|', round(m.d_off.abs().max(), 4),
              '| max |d slope|', round(m.d_slope.abs().max(), 5), '| 8 GP:', round(pg.d_off.abs().max(), 4), round(pg.d_slope.abs().max(), 5))


if __name__ == '__main__':
    main()
