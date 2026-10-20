"""2026-10-11 — Estimacion de tau (desviacion estandar del nivel log de degradacion por GP) por maxima
verosimilitud, para la opcion 3 jerarquica (ver pooled_hier_test_20261011.py). Covarianza OLS de la Ec. 1.

A) Dos etapas (meta-analisis de efectos aleatorios): del ajuste libre se toma theta_g (log del nivel de
   degradacion del GP) y su error estandar (inversa de J'J del GLS, que ya esta blanqueado); luego
   theta_g ~ N(mu, se_g^2 + tau^2) y se maximiza en (mu, tau) por ML y por REML.
B) Laplace: verosimilitud marginal aproximada del modelo conjunto completo en una grilla de tau,
   log L(tau) = -1/2 S(tau) - nG log tau - 1/2 log det(J_theta' J_theta), con S la suma de cuadrados
   penalizada en el optimo.
Escribe tau_ml_20261011.csv (grilla de B) e imprime las pendientes de los 8 GP con el tau de ML."""
import os, importlib.util, inspect, textwrap
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', os.path.join(HERE, 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
src = textwrap.dedent(inspect.getsource(P.fit_pooled))
src = src.replace("def fit_pooled(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False):",
                  "def fit_h(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False, tau_lev=None):")
src = src.replace("        out.append(tilt / tau_t)\n",
                  "        out.append(tilt / tau_t)\n        if kind == 'sl' and tau_lev:\n"
                  "            lev = p[STEPS:STEPS + nG]; out.append((lev - lev.mean()) / tau_lev)\n")
src = src.replace("    return {k: float(mean(sol.x, a, *k)) for k in cells}, a",
                  "    return {k: float(mean(sol.x, a, *k)) for k in cells}, a, sol, G")
ns = dict(vars(P)); exec(src, ns); fit_h = ns['fit_h']; STEPS = P.STEPS

dh = P.hist_laps_A()
H = {g: P.fit_gp(dh, g) for g in P.MAP if g in set(dh.GP)}
GPS = [g for g in P.MAP if g in H]; nG = len(GPS)
cfg = P.OPTIONS['opcion3']['sl']

# ---- A) dos etapas
s0, _, sol, G = fit_h(H, GPS, 'sl', **cfg)
J = sol.jac; C = np.linalg.pinv(J.T @ J)
th = sol.x[STEPS:STEPS + nG]; se = np.sqrt(np.diag(C)[STEPS:STEPS + nG])
T = pd.DataFrame(dict(GP=[g[:-11] for g in GPS], theta=th, nivel=np.exp(th), se_log=se)).sort_values('theta')
print(T.round(3).to_string(index=False))

def nll(tau, reml=False):
    w = 1 / (se ** 2 + tau ** 2); mu = np.sum(w * th) / np.sum(w)
    v = -0.5 * np.sum(np.log(w) - w * (th - mu) ** 2)
    return -(-v - (0.5 * np.log(np.sum(w)) if reml else 0)) * -1 if False else (v + (0.5 * np.log(np.sum(w)) if reml else 0))
for lab, reml in [('ML', False), ('REML', True)]:
    r = minimize_scalar(lambda t: nll(t, reml), bounds=(1e-3, 5), method='bounded')
    print(f'A) {lab}: tau = {r.x:.3f}')
tauA = minimize_scalar(lambda t: nll(t, False), bounds=(1e-3, 5), method='bounded').x
# IC aproximado por perfil (cambio de 1.92 en log-verosimilitud)
grid = np.linspace(0.05, 3, 300); ll = np.array([-nll(t) for t in grid]); ok = grid[ll >= ll.max() - 1.92]
print(f'   IC 95 % perfil (ML): [{ok.min():.2f}, {ok.max():.2f}]')

# ---- B) Laplace sobre el modelo completo
rows = []
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    s, _, sol, G = fit_h(H, GPS, 'sl', **cfg, tau_lev=tau)
    Jt = sol.jac[:, STEPS:STEPS + nG]
    S = 2 * sol.cost
    _, logdet = np.linalg.slogdet(Jt.T @ Jt)
    rows.append(dict(tau=tau, logL=-0.5 * S - nG * np.log(tau) - 0.5 * logdet,
                     saudi_C3=s[('Saudi Arabian Grand Prix', 'C3')]))
B = pd.DataFrame(rows); B.to_csv(os.path.join(HERE, 'tau_ml_20261011.csv'), index=False)
print(B.round(4).to_string(index=False))
tauB = B.tau[B.logL.idxmax()]
print(f'B) Laplace: tau = {tauB}')

for tau in sorted({round(tauA, 2), float(tauB)}):
    s, *_ = fit_h(H, GPS, 'sl', **cfg, tau_lev=tau)
    print(f'\npendientes con tau = {tau}:')
    for g in P.PAPER:
        print(f'  {g[:-11]:15s}', {c: round(s[(g, c)], 4) for c in P.MAP[g]})
