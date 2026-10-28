"""2026-10-28 — Fig. 4 y Ec. 2 con la definicion v10 (sin pasos por el pit lane, factor TyreLife 2021 = 0.740).

Correr desde la raiz del repo:
    F1_TYRELIFE_2021=v10 python results/driver_coef_v10_20261028/run_20261028.py v10
    python results/driver_coef_v10_20261028/run_20261028.py control   # sin parche, factor legacy: debe reproducir
                                                                     # results/driver_coef_study_20261010/
Corre, en la subcarpeta pedida, copias sin cambios de race_coefs_20261010 -> eq2_variants_20261010 ->
fig4_mse_by_gp_20261014 (que usa eq2_fallbacks_20261014/20261010). Las salidas quedan en esa subcarpeta.
"""
import os, sys, runpy
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1]
assert mode in ('control', 'v10')
assert os.environ.get('F1_TYRELIFE_2021', 'legacy') == ('v10' if mode == 'v10' else 'legacy')
import config, compat; compat.setup_all()
import models.regressions as R
if mode == 'v10':
    sys.path.insert(0, os.path.join(ROOT, 'results', 'sim_v10_20261027'))
    from common_v10_20261027 import patch_regressions
    patch_regressions(R)
D = os.path.join(HERE, mode)
for s in ('race_coefs_20261010.py', 'eq2_variants_20261010.py', 'fig4_mse_by_gp_20261014.py'):
    print('=====', mode, s, flush=True)
    runpy.run_path(os.path.join(D, s), run_name='__main__')
