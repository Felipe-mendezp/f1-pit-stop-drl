"""2026-10-28 — Efecto de v10d (v10c + Belgica y Monaco 2022) sobre los coeficientes de piloto del simulador
(Ec. 2 para 2024, 8 GP; frente a v10c) y sobre la Fig. 4. Desde la raiz del repo.
"""
import os, sys, importlib.util
import pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
from environment.f1_env_all_drivers import ALL_DRIVERS
from build_simulation_v5_20261010 import PAPER_GPS, SEAT
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for v in ('v10c', 'v10d'):
    s = importlib.util.spec_from_file_location('eq2_' + v, os.path.join(HERE, v, 'eq2_fallbacks_20261014.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); E = m.Eq2()
    for gp in PAPER_GPS:
        for n in ALL_DRIVERS:
            d = n.replace('Driver_', '')
            if d != 'VER':
                rows.append(dict(v=v, GP=gp, driver=d, coef=E.predict(2024, gp, SEAT.get((gp, d), d))['pred']))
D = pd.DataFrame(rows).pivot_table(index=['GP', 'driver'], columns='v', values='coef').reset_index()
D['dif'] = D.v10d - D.v10c
D.to_csv(os.path.join(HERE, 'compare_v10d_sim_drivers_20261028.csv'), index=False)
print(D.groupby('GP').dif.agg(lambda x: x.abs().max()).round(4).to_string())
print('media |dif| %.4f max %.4f' % (D.dif.abs().mean(), D.dif.abs().max()))
print(D.reindex(D.dif.abs().sort_values().index).tail(8).round(4).to_string(index=False))
a = pd.read_csv(os.path.join(HERE, 'v10c', 'fig4_mse_by_gp_20261014.csv')); b = pd.read_csv(os.path.join(HERE, 'v10d', 'fig4_mse_by_gp_20261014.csv'))
f = a.merge(b, on='GP', suffixes=('_v10c', '_v10d'))
f['d_with'] = f.with_corr_v10d - f.with_corr_v10c; f['d_without'] = f.without_corr_v10d - f.without_corr_v10c
print(f[['GP', 'with_corr_v10c', 'with_corr_v10d', 'without_corr_v10c', 'without_corr_v10d']][(f.d_with.abs() > 5e-4) | (f.d_without.abs() > 5e-4)].round(3).to_string(index=False))
