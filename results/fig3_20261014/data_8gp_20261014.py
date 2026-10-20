"""2026-10-14 — Coeficientes de neumatico de los 8 GP en tres etapas (para la pagina de comparacion):
(1) Ec. 1 por circuito con todas las vueltas, (2) tras el IQR iterativo, (3) modelo conjunto opcion3h.
Datos 2021-23 con filtro A. Offsets relativos al compuesto mas duro con datos del GP (Dutch: C2). Salida: results/fig3_20261014/coefs_8gp.json"""
import os, sys, io, json, contextlib, importlib.util
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', 'results/tyre_coef_study_20260930/pooled_20261004.py')
with contextlib.redirect_stdout(io.StringIO()):
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
raw = pd.concat(R.load_data()[:3]); wet = P.wet_editions(raw)
with contextlib.redirect_stdout(io.StringIO()):
    lw = R.all_laps(outliers=True)
lw = lw[[(y, g) not in wet for y, g in zip(lw.Year, lw.GP)]]
dh = P.hist_laps_A()
J = pd.read_csv('results/tyre_coef_study_20260930/pooled_hier_20261011/coefs_opcion3h.csv')
NL = {'Bahrain': 57, 'Saudi Arabian': 50, 'Miami': 57, 'Hungarian': 70, 'Belgian': 44, 'Dutch': 72, 'Singapore': 62, 'United States': 56}
out = []
for g in P.PAPER:
    comps = P.MAP[g]
    d = lw[(lw.GP == g) & (lw.TrackStatus == 1.0)]
    with contextlib.redirect_stdout(io.StringIO()):
        m, _ = R.calibrar_regresion_gp(d)
    pr, bse = m.params, m.bse
    present = sorted(d.Compound_Detail.unique())
    ref = [c for c in present if not any(f'[T.{c}]' in p for p in pr.index if 'Compound_Detail, Treatment' in p)][0]
    def st1(c):
        on = f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"; sn = f"TyreLife:C(Compound_Detail)[{c}]"
        o = 0.0 if c == ref else pr.get(on, np.nan); s = pr.get(sn, np.nan)
        return o, s
    H = P.fit_gp(dh, g); b = dict(zip(H['kinds'], H['b']))
    def st2(c):
        o = 0.0 if c == H['ref'] else b.get(('off', c), np.nan); return o, b.get(('sl', c), np.nan)
    x = J[J.GP == g].set_index('compound')
    hard = [c for c in comps if H['stints'].get(c, 0) > 0][0]   # base: compuesto mas duro con datos
    stages = []
    for name, f in (('raw', st1), ('clean', st2), ('joint', lambda c: (x.offset_vs_ref[c], x.slope[c]))):
        vals = {c: f(c) for c in comps}
        h = vals[hard][0]
        stages.append(dict(stage=name, coefs={c: dict(offset=None if not np.isfinite(vals[c][0] - h) else round(float(vals[c][0] - h), 4),
                                                       slope=None if not np.isfinite(vals[c][1]) else round(float(vals[c][1]), 4)) for c in comps}))
    out.append(dict(gp=g.replace(' Grand Prix', ''), base=hard, laps=NL[g.replace(' Grand Prix', '')], compounds=comps,
                    stints={c: int(H['stints'].get(c, 0)) for c in comps},
                    n_raw=int(len(d)), n_clean=int(sum(H['laps'].values())), stages=stages))
json.dump(out, open(os.path.join(HERE, 'coefs_8gp.json'), 'w'), indent=1)
for r in out:
    print(r['gp'], r['n_raw'], r['n_clean'], r['stints'], [(s['stage'], {c: (v['offset'], v['slope']) for c, v in s['coefs'].items()}) for s in r['stages']])
