"""2026-10-28 — Cadena v10eG y v10eGQ (edad efectiva) frente a v10d: neumaticos del simulador (8 GP), resto de pickles,
pilotos (Ec. 2) y tau. Desde la raiz del repo. Salida: tyres_8gp.csv, pickles.csv, drivers.csv, compare_chain.txt
"""
import os, sys, pickle, importlib.util
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
from environment.f1_env_all_drivers import ALL_DRIVERS
from build_simulation_v5_20261010 import PAPER_GPS, SEAT
HERE = os.path.dirname(os.path.abspath(__file__))
V = {'v10d': 'sim_v10d_20261028', 'v10eG': 'sim_v10eG_20261028', 'v10eGQ': 'sim_v10eGQ_20261028'}
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
pd.set_option('display.width', 220)

def main():   # los pickles de statsmodels (patsy) no cargan a nivel de modulo
    # neumaticos del simulador (8 GP) y mundo 2024
    for f in ('coefs_opcion3h.csv', 'coefs_opcion3h_2021_24.csv'):
        T = None
        for v, d in V.items():
            x = pd.read_csv(os.path.join(ROOT, 'results', d, 'tyre_v10', f))[['GP', 'compound', 'offset_vs_ref', 'slope']]
            x = x.rename(columns={'offset_vs_ref': f'off_{v}', 'slope': f'sl_{v}'})
            T = x if T is None else T.merge(x, on=['GP', 'compound'])
        T = T[T.GP.isin(PAPER_GPS)]
        T.to_csv(os.path.join(HERE, 'tyres_8gp' + ('_mundo2024' if '2021_24' in f else '') + '.csv'), index=False)
        say(f'\n== {f} (8 GP): desplazamiento frente al compuesto de referencia (s) y pendiente (s/vuelta)')
        say(T.round(3).to_string(index=False))
        for v in ('v10eG', 'v10eGQ'):
            say(f'  {v}: pendiente media {T[f"sl_{v}"].mean():.4f} vs v10d {T.sl_v10d.mean():.4f} '
                f'(cambio relativo medio {100 * (T[f"sl_{v}"] / T.sl_v10d - 1).mean():+.1f}%); max |d desplazamiento| '
                f'{(T[f"off_{v}"] - T.off_v10d).abs().max():.3f}')
    for v, d in V.items():
        say(f'tau {v}:', open(os.path.join(ROOT, 'results', d, 'tyre_v10', 'tau.txt')).read().replace('\n', ' '))

    # demas pickles (sin pilotos)
    def coefs(o):
        if hasattr(o, 'feature_names_'):
            return dict(zip(o.feature_names_, np.asarray(o.coef_, float))) | {'Intercept': float(o.intercept_)}
        p = getattr(o, 'params', None)
        if p is not None:
            return {k: float(v) for k, v in p.items()}
        return {'valor': float(o)} if np.isscalar(o) else {}
    rows = []
    base = os.path.join(ROOT, 'results', V['v10d'], 'artifacts')
    for gp in PAPER_GPS:
        for f in sorted(os.listdir(os.path.join(base, gp))):
            a = coefs(pickle.load(open(os.path.join(base, gp, f), 'rb')))
            for v in ('v10eG', 'v10eGQ'):
                b = coefs(pickle.load(open(os.path.join(ROOT, 'results', V[v], 'artifacts', gp, f), 'rb')))
                for k in a:
                    if not k.startswith('C(Driver') and k in b:
                        rows.append(dict(v=v, GP=gp, file=f, coef=k, v10d=a[k], new=b[k], dif=b[k] - a[k]))
    P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, 'pickles.csv'), index=False)
    say('\n== resto de modelos: max |dif| por archivo (sin pilotos)')
    say(P.assign(a=P.dif.abs()).groupby(['file', 'v']).a.max().unstack().round(4).to_string())
    L = P[(P.file == 'rival_pitstop_logit.pkl') & (P.GP == PAPER_GPS[0])]
    say('\nLogitStop (coeficientes de edad y SC/VSC):')
    say(L[L.coef.str.contains('TyreLife|SC|VSC|Intercept')].pivot_table(index='coef', columns='v', values=['v10d', 'new']).round(4).to_string())

    # pilotos (Ec. 2, 2024)
    rows = []
    for v in ('v10d', 'v10eG', 'v10eGQ'):
        s = importlib.util.spec_from_file_location('eq2_' + v, os.path.join(ROOT, 'results', 'driver_coef_v10_20261028', v, 'eq2_fallbacks_20261014.py'))
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); E = m.Eq2()
        for gp in PAPER_GPS:
            for n in ALL_DRIVERS:
                dd = n.replace('Driver_', '')
                if dd != 'VER':
                    rows.append(dict(v=v, GP=gp, driver=dd, coef=E.predict(2024, gp, SEAT.get((gp, dd), dd))['pred']))
    D = pd.DataFrame(rows).pivot_table(index=['GP', 'driver'], columns='v', values='coef').reset_index()
    D.to_csv(os.path.join(HERE, 'drivers.csv'), index=False)
    for v in ('v10eG', 'v10eGQ'):
        d = (D[v] - D.v10d).abs()
        say(f'pilotos {v} vs v10d: media |dif| {d.mean():.4f}, p95 {d.quantile(.95):.4f}, max {d.max():.4f} s/vuelta')
    for v in ('v10d', 'v10eG', 'v10eGQ'):
        f = pd.read_csv(os.path.join(ROOT, 'results', 'driver_coef_v10_20261028', v, 'fig4_mse_by_gp_20261014.csv'))
        say(f'Fig. 4 {v}: media con correccion {f.with_corr.mean():.3f}, sin {f.without_corr.mean():.3f}')
    open(os.path.join(HERE, 'compare_chain.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
