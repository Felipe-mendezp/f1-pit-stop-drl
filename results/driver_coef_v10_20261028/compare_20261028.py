"""2026-10-28 — Compara la Fig. 4 y los coeficientes de piloto del simulador (Ec. 2, 2024, 8 GP) entre:
  publicado = results/driver_coef_study_20261010/  (el del paper y del simulador v6/v7/v10)
  control   = control/  (copia re-corrida, factor legacy, sin parche: debe ser igual a publicado)
  v10       = v10/      (sin pasos por el pit lane, factor 2021 = 0.740)
Correr desde la raiz del repo. Salidas: compare_fig4_20261028.csv, compare_sim_drivers_20261028.csv.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()
from environment.f1_env_all_drivers import ALL_DRIVERS
from build_simulation_v5_20261010 import PAPER_GPS, SEAT
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {'publicado': os.path.join(ROOT, 'results', 'driver_coef_study_20261010'),
        'control': os.path.join(HERE, 'control'), 'v10': os.path.join(HERE, 'v10')}

# Fig. 4
F = None
for k, d in DIRS.items():
    g = pd.read_csv(os.path.join(d, 'fig4_mse_by_gp_20261014.csv'))[['GP', 'Round', 'n', 'with_corr', 'without_corr']]
    g = g.rename(columns={c: f'{c}_{k}' for c in ('n', 'with_corr', 'without_corr')})
    F = g if F is None else F.merge(g, on=['GP', 'Round'], how='outer')
F = F.sort_values('Round'); F.to_csv(os.path.join(HERE, 'compare_fig4_20261028.csv'), index=False)
pd.set_option('display.width', 250)
print(F.round(3).to_string(index=False))
for k in DIRS:
    print(f'{k}: media con {F["with_corr_"+k].mean():.3f} sin {F["without_corr_"+k].mean():.3f}; '
          f'corr empeora en {list(F[F["with_corr_"+k] > F["without_corr_"+k]].GP)}')

# coeficientes de piloto del simulador (Ec. 2 para 2024)
rows = []
for k, d in DIRS.items():
    s = importlib.util.spec_from_file_location('eq2_' + k, os.path.join(d, 'eq2_fallbacks_20261014.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); E = m.Eq2()
    for gp in PAPER_GPS:
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            if drv == 'VER':
                continue
            rows.append(dict(src=k, GP=gp, driver=drv, coef=E.predict(2024, gp, SEAT.get((gp, drv), drv))['pred']))
D = pd.DataFrame(rows).pivot_table(index=['GP', 'driver'], columns='src', values='coef').reset_index()
D['d_control'] = D.control - D.publicado; D['d_v10'] = D.v10 - D.publicado
D.to_csv(os.path.join(HERE, 'compare_sim_drivers_20261028.csv'), index=False)
print('max |control - publicado| =', D.d_control.abs().max())
print(D.groupby('GP').d_v10.agg(lambda x: x.abs().max()).rename('max |v10 - publicado|').round(4).to_string())
print('v10 - publicado: media |dif| %.4f, p95 %.4f, max %.4f s/vuelta' % (D.d_v10.abs().mean(), D.d_v10.abs().quantile(.95), D.d_v10.abs().max()))
