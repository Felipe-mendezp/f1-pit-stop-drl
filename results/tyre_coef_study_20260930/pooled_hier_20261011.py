"""2026-10-11 — Opcion 3 jerarquica ("opcion3h"): igual que la opcion 3 de pooled_20261004.py, pero el nivel
log de degradacion de cada GP (theta_g) es un efecto aleatorio, theta_g ~ N(mu, tau^2), con
tau = 0.743 estimado por maxima verosimilitud (tau_ml_20261011.py). Los offsets no cambian (mismo ajuste).
Motivo: Saudi 2021-23 no identifica la degradacion (corr(TyreLife, LapNumber) = 0.82, se del nivel log 1.74)
y la opcion 3 la deja en ~0.

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/pooled_hier_20261011.py
Salida: pooled_hier_20261011/coefs_opcion3h.csv (mismo formato que pooled_20261004/coefs_opcion3.csv) y
diff_vs_opcion3.csv."""
import os, importlib.util, inspect, textwrap
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.743
spec = importlib.util.spec_from_file_location('p0', os.path.join(HERE, 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
src = textwrap.dedent(inspect.getsource(P.fit_pooled))
src = src.replace("def fit_pooled(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False):",
                  "def fit_h(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False, tau_lev=None):")
src = src.replace("        out.append(tilt / tau_t)\n",
                  "        out.append(tilt / tau_t)\n        if kind == 'sl' and tau_lev:\n"
                  "            lev = p[STEPS:STEPS + nG]; out.append((lev - lev.mean()) / tau_lev)\n")
ns = dict(vars(P)); exec(src, ns); fit_h = ns['fit_h']
OUT = os.path.join(HERE, 'pooled_hier_20261011'); os.makedirs(OUT, exist_ok=True)

dh = P.hist_laps_A()
H = {g: P.fit_gp(dh, g) for g in P.MAP if g in set(dh.GP)}
GPS = [g for g in P.MAP if g in H]
cfg = P.OPTIONS['opcion3']
o, _ = P.fit_pooled(H, GPS, 'off', **cfg['off'])
s, _ = fit_h(H, GPS, 'sl', **cfg['sl'], tau_lev=TAU)
rows = [dict(GP=g, ref=H[g]['ref'], compound=c, offset_vs_ref=o[(g, c)] - o[(g, H[g]['ref'])], slope=s[(g, c)],
             stints_2021_23=H[g]['stints'].get(c, 0)) for g in GPS for c in P.MAP[g]]
N = pd.DataFrame(rows); N.to_csv(os.path.join(OUT, 'coefs_opcion3h.csv'), index=False)
O = pd.read_csv(os.path.join(HERE, 'pooled_20261004', 'coefs_opcion3.csv'))
D = O.merge(N, on=['GP', 'ref', 'compound'], suffixes=('_op3', '_op3h'))
D['d_offset'] = D.offset_vs_ref_op3h - D.offset_vs_ref_op3; D['d_slope'] = D.slope_op3h - D.slope_op3
D.to_csv(os.path.join(OUT, 'diff_vs_opcion3.csv'), index=False)
pd.set_option('display.width', 200)
print('max |d_offset| en todos los GP:', D.d_offset.abs().max().round(6))
print(D[D.GP.isin(P.PAPER)][['GP', 'compound', 'slope_op3', 'slope_op3h', 'd_slope']].round(5).to_string(index=False))
