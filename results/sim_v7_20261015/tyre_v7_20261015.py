"""2026-10-15 — Coeficientes de neumatico del simulador v7 (Fase 2 de docs/DISENO_TYRELIFE_2021_20261015.md).

Misma cadena que v6, con la vida del neumatico de 2021 segun F1_TYRELIFE_2021 (v7 = factor unico 0.716):
  1. Ec. 1 por GP con 2021-23, filtro (A) e IQR (pooled_20261004.hist_laps_A / fit_gp).
  2. Offsets: modelo conjunto opcion3 (pooled_20261004.fit_pooled).
  3. Pendientes: opcion3h, nivel log de degradacion por GP con efecto aleatorio; tau re-estimado por ML en dos
     etapas (mismo metodo que dio tau = 0.743 en v6: world2024h_20261012.tau_ml).
  4. "Mundo 2024": lo mismo con 2021-24 (world2024h_20261012), para evaluar en neumaticos 2024.
Control: con F1_TYRELIFE_2021=legacy debe reproducir pooled_hier_20261011/coefs_opcion3h.csv (tau 0.743) y
world2024h_20261012/coefs_opcion3h_2021_24.csv.

Correr desde la raiz del repo:  F1_TYRELIFE_2021=v7 python results/sim_v7_20261015/tyre_v7_20261015.py
Salida: results/sim_v7_20261015/tyre_<tag>/{coefs_opcion3h.csv, coefs_opcion3h_2021_24.csv, tau.txt}
"""
import importlib.util, os, sys
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import pandas as pd

TAG = os.environ.get('F1_TYRELIFE_2021', 'legacy')
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(ROOT, 'results', 'tyre_coef_study_20260930')
OUT = os.path.join(HERE, f'tyre_{TAG}'); os.makedirs(OUT, exist_ok=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


W = load('world2024h', os.path.join(STUDY, 'world2024h_20261012.py'))
P = W.P


def coefs(d, GPS, H):
    cfg = P.OPTIONS['opcion3']
    tau = W.tau_ml(H, GPS, cfg['sl'])
    o, _ = P.fit_pooled(H, GPS, 'off', **cfg['off'])
    s, *_ = W.fit_h(H, GPS, 'sl', **cfg['sl'], tau_lev=tau)
    rows = [dict(GP=g, ref=H[g]['ref'], compound=c, offset_vs_ref=o[(g, c)] - o[(g, H[g]['ref'])], slope=s[(g, c)])
            for g in GPS for c in P.MAP[g]]
    return pd.DataFrame(rows), tau


def main():
    dh, d24 = P.hist_laps_A(), P.laps_2024_A()
    print('vueltas 2021-23 (filtro A):', len(dh), '| TyreLife medio 2021:', round(dh[dh.Year == 2021].TyreLife.mean(), 3))
    H = {g: P.fit_gp(dh, g) for g in P.MAP if g in set(dh.GP)}
    GPS = [g for g in P.MAP if g in H]
    C, tau = coefs(dh, GPS, H)
    C.to_csv(os.path.join(OUT, 'coefs_opcion3h.csv'), index=False)
    d = pd.concat([dh, d24], ignore_index=True, sort=False)
    H4 = {g: P.fit_gp(d, g) for g in GPS}
    C4, tau4 = coefs(d, GPS, H4)
    C4.to_csv(os.path.join(OUT, 'coefs_opcion3h_2021_24.csv'), index=False)
    with open(os.path.join(OUT, 'tau.txt'), 'w') as f:
        f.write(f'tau_2021_23 {tau:.6f}\ntau_2021_24 {tau4:.6f}\n')
    print(f'{TAG}: tau 2021-23 = {tau:.4f} | 2021-24 = {tau4:.4f}')
    for new, ref in [(C, os.path.join(STUDY, 'pooled_hier_20261011', 'coefs_opcion3h.csv')),
                     (C4, os.path.join(STUDY, 'world2024h_20261012', 'coefs_opcion3h_2021_24.csv'))]:
        old = pd.read_csv(ref)
        m = old.merge(new, on=['GP', 'ref', 'compound'], suffixes=('_v6', '_new'))
        print(os.path.basename(ref), 'filas', len(m), '/', len(old),
              '| max |d offset|', round((m.offset_vs_ref_new - m.offset_vs_ref_v6).abs().max(), 6),
              '| max |d slope|', round((m.slope_new - m.slope_v6).abs().max(), 6))


if __name__ == '__main__':
    main()
