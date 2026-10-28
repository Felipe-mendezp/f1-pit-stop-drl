"""2026-10-27 — Tabla de la Ec. 1 de US antes/despues de outliers y figura de vueltas outlier con la calibracion v10
(sin pasos por el pit lane, vida del neumatico de 2021 x 0.740). Copia de results/outliers_us_v7_20261019/run_v7_20261019.py.
    F1_TYRELIFE_2021=v10 python results/sim_v10_20261027/outliers_us_v10_20261027.py   -> results/sim_v10_20261027/outliers_us_v10/
"""
import os, sys
HERE0 = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'src'), HERE0]
assert os.environ.get('F1_TYRELIFE_2021') == 'v10eGQ'  # v10e
import config, compat; compat.setup_all()
import models.regressions as R
import common_v10_20261027 as CV
CV.patch_regressions(R)
HERE = os.path.join(HERE0, 'outliers_us_v10'); os.makedirs(HERE, exist_ok=True)
for name in ('us_outliers_20261014.py', 'table_tex_20261014.py'):
    path = os.path.join('results', 'outliers_us_20261014', name)
    src = open(path).read()
    old = "HERE = os.path.dirname(os.path.abspath(__file__))"
    assert src.count(old) == 1
    exec(compile(src.replace(old, f"HERE = {HERE!r}"), path, 'exec'), {'__name__': '__main__', '__file__': path})
