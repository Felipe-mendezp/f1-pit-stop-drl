"""2026-10-10 — Prueba de humo del arbol simulation_v5_20261010 (etapa B).

Para cada GP y cada uno de los 20 pilotos, construye el estado de la vuelta 1 con initial_lap_v3 (como
el entorno) y verifica que la prediccion del arbol nuevo menos la del arbol base sea exactamente
coef_eq2 - coef_hist (driver_coef_audit.csv), y que nada mas cambie (mismo compuesto, misma vuelta).
Correr desde la raiz del repo.
"""
import os, sys, pickle
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import compat; compat.setup_all()
import pandas as pd
from utils import initial_lap_v3
from environment.f1_env_all_drivers import ALL_DRIVERS

M = os.path.join(ROOT, 'trained_models')
NEW, BASE = os.path.join(M, 'simulation_v5_20261010'), os.path.join(M, 'simulation_yfv4_opcion3_20261007')
A = pd.read_csv(os.path.join(NEW, 'driver_coef_audit.csv')).set_index(['GP', 'driver'])
worst = 0.0
for gp in A.index.get_level_values(0).unique():
    fn, fb = (pickle.load(open(os.path.join(t, gp, 'laptime_clear.pkl'), 'rb')) for t in (NEW, BASE))
    for f in os.listdir(os.path.join(BASE, gp)):
        if f != 'laptime_clear.pkl':
            assert os.path.realpath(os.path.join(NEW, gp, f)) == os.path.realpath(os.path.join(BASE, gp, f)), (gp, f)
    for comp in (2, 3, 4):
        for name in ALL_DRIVERS:
            d = name.replace('Driver_', ''); r = A.loc[(gp, d)]
            sn, sb = initial_lap_v3(comp, name, fn), initial_lap_v3(comp, name, fb)
            delta = fn.predict_single(sn) - fb.predict_single(sb)
            err = abs(delta - (r.coef_eq2 - r.coef_hist))
            worst = max(worst, err)
            assert err < 1e-3, (gp, d, comp, delta, r.coef_eq2 - r.coef_hist)
    print(f'{gp}: OK')
print(f'max |error| = {worst:.2e} s; demas archivos = links a la base')
