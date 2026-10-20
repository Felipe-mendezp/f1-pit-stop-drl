"""2026-10-12 — Coeficientes de neumatico "mundo 2024" jerarquicos: opcion3h re-estimada con 2021-2024.

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/world2024h_20261012.py
Salida (nada fuera de esta carpeta): results/tyre_coef_study_20260930/world2024h_20261012/

Mismos datos que world2024_20261010.py (Ec. 1 de cada GP con 2021-2024, 2024 como un anio mas, filtro (A))
y mismo metodo que la creencia opcion3h (pooled_hier_20261011.py): offsets del modelo conjunto opcion3 y
pendientes con el nivel log de degradacion por GP como efecto aleatorio. tau se re-estima por maxima
verosimilitud con estos datos (dos etapas, como el tau = 0.743 de tau_ml_20261011.py, metodo A).
Decidido por Charles 2026-10-12: creencia = opcion3h 2021-23 (agentes y QMIP), mundo = opcion3h 2021-24.
"""
import importlib.util, inspect, os, sys, textwrap
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('pooled_20261004', os.path.join(HERE, 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
OUT = os.path.join(HERE, 'world2024h_20261012'); os.makedirs(OUT, exist_ok=True)

# fit_pooled con penalizacion jerarquica del nivel (mismo parche que pooled_hier / tau_ml)
src = textwrap.dedent(inspect.getsource(P.fit_pooled))
src = src.replace("def fit_pooled(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False):",
                  "def fit_h(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False, tau_lev=None):")
src = src.replace("        out.append(tilt / tau_t)\n",
                  "        out.append(tilt / tau_t)\n        if kind == 'sl' and tau_lev:\n"
                  "            lev = p[STEPS:STEPS + nG]; out.append((lev - lev.mean()) / tau_lev)\n")
src = src.replace("    return {k: float(mean(sol.x, a, *k)) for k in cells}, a",
                  "    return {k: float(mean(sol.x, a, *k)) for k in cells}, a, sol")
ns = dict(vars(P)); exec(src, ns); fit_h = ns['fit_h']


def tau_ml(H, GPS, cfg):
    """Dos etapas: theta_g y su error estandar del ajuste libre; theta_g ~ N(mu, se_g^2 + tau^2), ML en tau."""
    nG = len(GPS)
    _, _, sol = fit_h(H, GPS, 'sl', **cfg)
    C = np.linalg.pinv(sol.jac.T @ sol.jac)
    th = sol.x[P.STEPS:P.STEPS + nG]; se = np.sqrt(np.diag(C)[P.STEPS:P.STEPS + nG])

    def nll(tau):
        w = 1 / (se ** 2 + tau ** 2); mu = np.sum(w * th) / np.sum(w)
        return -0.5 * np.sum(np.log(w) - w * (th - mu) ** 2)
    return minimize_scalar(nll, bounds=(1e-3, 5), method='bounded').x


def main():
    dh, d24 = P.hist_laps_A(), P.laps_2024_A()
    d = pd.concat([dh, d24], ignore_index=True, sort=False)
    print('vueltas por anio (filtro A):', d.groupby('Year').size().to_dict())
    H = {g: P.fit_gp(d, g) for g in P.MAP if g in set(dh.GP)}   # mismos GP que opcion3 (con historia 2021-23)
    GPS = [g for g in P.MAP if g in H]
    cfg = P.OPTIONS['opcion3']
    tau = tau_ml(H, GPS, cfg['sl'])
    # control: el mismo calculo con 2021-23 debe reproducir el tau de la creencia (0.743)
    H0 = {g: P.fit_gp(dh, g) for g in GPS}
    tau0 = tau_ml(H0, GPS, cfg['sl'])
    print(f'tau ML: 2021-23 = {tau0:.3f} (creencia usa 0.743) | 2021-24 = {tau:.3f}')
    o, _ = P.fit_pooled(H, GPS, 'off', **cfg['off'])
    s, *_ = fit_h(H, GPS, 'sl', **cfg['sl'], tau_lev=tau)
    rows = [dict(GP=g, ref=H[g]['ref'], compound=c, offset_vs_ref=o[(g, c)] - o[(g, H[g]['ref'])],
                 slope=s[(g, c)], stints_2021_24=H[g]['stints'].get(c, 0)) for g in GPS for c in P.MAP[g]]
    new = pd.DataFrame(rows); new.to_csv(os.path.join(OUT, 'coefs_opcion3h_2021_24.csv'), index=False)
    with open(os.path.join(OUT, 'tau.txt'), 'w') as f:
        f.write(f'tau_ml_2021_24 {tau:.6f}\ntau_ml_2021_23_control {tau0:.6f}\n')

    # comparacion de pendientes y fijos (vs compuesto medio) en los 8 GP del paper
    tabs = [('creencia_op3h_2123', os.path.join(HERE, 'pooled_hier_20261011', 'coefs_opcion3h.csv')),
            ('mundo_op3_2124', os.path.join(HERE, 'world2024_20261010', 'coefs_opcion3_2021_24.csv')),
            ('mundo_op3h_2124', None)]
    cmp = []
    for lab, path in tabs:
        t = new if path is None else pd.read_csv(path)
        for g in P.PAPER:
            x = t[t.GP == g].set_index('compound'); m = P.MAP[g][1]
            for c in P.MAP[g]:
                cmp.append(dict(GP=g.replace(' Grand Prix', ''), compound=c, version=lab,
                                fijo_vs_medio=x.offset_vs_ref[c] - x.offset_vs_ref[m], pendiente=x.slope[c]))
    C = pd.DataFrame(cmp).pivot_table(index=['GP', 'compound'], columns='version',
                                      values=['pendiente', 'fijo_vs_medio'])
    C.to_csv(os.path.join(OUT, 'compare_8gp.csv'))
    pd.set_option('display.width', 220)
    print(C.round(4).to_string())


if __name__ == '__main__':
    main()
