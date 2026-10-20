"""2026-10-11 — Prueba: opcion 3 con nivel de degradacion por GP jerarquico.

En pooled_20261004.fit_pooled el nivel de degradacion de cada GP (theta_g) es un efecto fijo sin
penalizacion: lo determina solo el GP, cualquiera sea su precision. Aqui se agrega la penalizacion
(theta_g - media(theta)) / tau_lev (efecto aleatorio en escala log) y se compara con covarianza OLS y
agrupada por stint. Solo pendientes; imprime la tabla de los 8 GP del paper. No escribe coeficientes."""
import os, sys, importlib.util, inspect, textwrap
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('pc', os.path.join(HERE, 'pooled_cluster_20261011.py'))
PC = importlib.util.module_from_spec(spec); spec.loader.exec_module(PC)
P = PC.P
src = textwrap.dedent(inspect.getsource(P.fit_pooled))
src = src.replace("def fit_pooled(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False):",
                  "def fit_pooled_h(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False, tau_lev=None):")
src = src.replace("        out.append(tilt / tau_t)\n",
                  "        out.append(tilt / tau_t)\n        if kind == 'sl' and tau_lev:\n"
                  "            lev = p[STEPS:STEPS + nG]; out.append((lev - lev.mean()) / tau_lev)\n")
ns = dict(vars(P)); exec(src, ns); fit_h = ns['fit_pooled_h']
dh = P.hist_laps_A()
cfg = P.OPTIONS['opcion3']['sl']
rows = []
for cov, fg in [('OLS', P.R and None), ('cluster', PC.fit_gp_cluster)]:
    if cov == 'OLS':
        spec2 = importlib.util.spec_from_file_location('p0', os.path.join(HERE, 'pooled_20261004.py'))
        P0 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(P0); fg = P0.fit_gp
    H = {g: fg(dh, g) for g in P.MAP if g in set(dh.GP)}
    GPS = [g for g in P.MAP if g in H]
    for tl in [None, 1.0, 0.5, 0.3]:
        s, _ = fit_h(H, GPS, 'sl', **cfg, tau_lev=tl)
        for g in P.PAPER:
            if g.startswith('Emilia'): continue
            rows.append(dict(cov=cov, tau_lev=tl or 'libre', GP=g[:-11], **{c: round(s[(g, c)], 4) for c in P.MAP[g]}))
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
for g, x in D.groupby('GP', sort=False):
    print(x.dropna(axis=1, how='all').to_string(index=False)); print()
D.to_csv(os.path.join(HERE, 'pooled_hier_test_20261011.csv'), index=False)
