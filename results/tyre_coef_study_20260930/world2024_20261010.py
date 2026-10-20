"""2026-10-10 — Coeficientes de neumatico "mundo 2024": opcion3 re-estimada con 2021-2024.

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/world2024_20261010.py
Salida (nada fuera de esta carpeta): results/tyre_coef_study_20260930/world2024_20261010/

Misma metodologia que opcion3 (pooled_20261004.py: Ec. 1 por GP + filtro (A) + modelo conjunto
monotono estructural, paso C4->C5 = paso C3->C4), pero la Ec. 1 de cada GP se ajusta con las
carreras 2021-2024: 2024 entra como un anio mas (su propio intercepto C(Year), mismas
transformaciones que laps_2024_A, filtro (A) aplicado a sus ediciones). Solo para EVALUAR
(Figs. 6-7): los agentes y la QMIP siguen creyendo los coeficientes opcion3 2021-23.
"""
import importlib.util, os, sys
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('pooled_20261004', os.path.join(HERE, 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
OUT = os.path.join(HERE, 'world2024_20261010'); os.makedirs(OUT, exist_ok=True)


def main():
    dh, d24 = P.hist_laps_A(), P.laps_2024_A()
    d = pd.concat([dh, d24], ignore_index=True, sort=False)
    print('vueltas por anio (filtro A):', d.groupby('Year').size().to_dict())
    H = {g: P.fit_gp(d, g) for g in P.MAP if g in set(dh.GP)}   # mismos GP que opcion3 (con historia 2021-23)
    GPS = [g for g in P.MAP if g in H]
    cfg = P.OPTIONS['opcion3']
    o, pat_o = P.fit_pooled(H, GPS, 'off', **cfg['off'])
    s, pat_s = P.fit_pooled(H, GPS, 'sl', **cfg['sl'])
    rows = [dict(GP=g, ref=H[g]['ref'], compound=c, offset_vs_ref=o[(g, c)] - o[(g, H[g]['ref'])],
                 slope=s[(g, c)], stints_2021_24=H[g]['stints'].get(c, 0)) for g in GPS for c in P.MAP[g]]
    new = pd.DataFrame(rows); new.to_csv(os.path.join(OUT, 'coefs_opcion3_2021_24.csv'), index=False)

    # comparacion con opcion3 2021-23 (8 GP del paper), offsets relativos al compuesto medio
    old = pd.read_csv(os.path.join(HERE, 'pooled_20261004', 'coefs_opcion3.csv'))
    cmp = []
    for g in P.PAPER:
        m = P.MAP[g][1]
        for lab, t in [('2021-23', old), ('2021-24', new)]:
            x = t[t.GP == g].set_index('compound')
            for c in P.MAP[g]:
                cmp.append(dict(GP=g.replace(' Grand Prix', ''), datos=lab, compound=c,
                                fijo_vs_medio=x.offset_vs_ref[c] - x.offset_vs_ref[m], pendiente=x.slope[c]))
    C = pd.DataFrame(cmp).pivot_table(index=['GP', 'compound'], columns='datos',
                                      values=['fijo_vs_medio', 'pendiente'])
    C.to_csv(os.path.join(OUT, 'compare_8gp.csv'))
    pd.set_option('display.width', 200)
    print(C.round(3).to_string())


if __name__ == '__main__':
    main()
