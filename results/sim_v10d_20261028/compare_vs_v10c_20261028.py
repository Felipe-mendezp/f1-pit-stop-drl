"""2026-10-28 — Artefactos v10d (esta carpeta) frente a v10c (results/sim_v10_20261028). Copia de compare_vs_v10_20261028.py.
Desde la raiz del repo. Salida: compare_vs_v10c_20261028.txt
"""
import os, sys, pickle
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
HERE = os.path.dirname(os.path.abspath(__file__)); OLD = os.path.join(ROOT, 'results', 'sim_v10_20261028')
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
def coefs(o):
    if hasattr(o, 'feature_names_'):
        return dict(zip(o.feature_names_, np.asarray(o.coef_, float))) | {'Intercept': float(o.intercept_)}
    p = getattr(o, 'params', None)
    if p is not None:
        return {k: float(v) for k, v in p.items()}
    if hasattr(o, 'coef_'):
        return {f'c{i}': float(v) for i, v in enumerate(np.ravel(o.coef_))} | {'i': float(np.ravel(getattr(o, 'intercept_', [0]))[0])}
    return {'valor': float(o)} if np.isscalar(o) else {}
def main():
    for gp in sorted(g for g in os.listdir(os.path.join(HERE, 'artifacts')) if os.path.isdir(os.path.join(HERE, 'artifacts', g))):
        for f in sorted(os.listdir(os.path.join(HERE, 'artifacts', gp))):
            a = pickle.load(open(os.path.join(OLD, 'artifacts', gp, f), 'rb')); b = pickle.load(open(os.path.join(HERE, 'artifacts', gp, f), 'rb'))
            ca, cb = coefs(a), coefs(b)
            ks = [k for k in set(ca) | set(cb) if not k.startswith('C(Driver')]
            d = {k: abs(cb.get(k, np.nan) - ca.get(k, np.nan)) for k in ks}
            k = max(d, key=lambda x: -1 if np.isnan(d[x]) else d[x]) if d else None
            say(f'{gp:26s} {f:28s} max |dif| (sin pilotos) {d[k] if k else np.nan:.4g} en {k}')
    a = pd.read_csv(os.path.join(OLD, 'yf_model_v10_20261027.csv')).set_index('GP'); b = pd.read_csv(os.path.join(HERE, 'yf_model_v10_20261027.csv')).set_index('GP')
    say('YF: max |dif| por columna:', (b - a).abs().max().round(4).to_dict())
    open(os.path.join(HERE, 'compare_vs_v10c_20261028.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
