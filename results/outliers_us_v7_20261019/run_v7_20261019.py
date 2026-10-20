"""2026-10-19 — Item 10 del plan: tabla de la Ec. 1 de US antes/despues de outliers y figura de vueltas outlier con la
calibracion v7 (vida del neumatico de 2021 x 0.716; correr con F1_TYRELIFE_2021=v7). Ejecuta, sin modificarlos,
results/outliers_us_20261014/us_outliers_20261014.py y table_tex_20261014.py con la salida en esta carpeta.
    F1_TYRELIFE_2021=v7 python results/outliers_us_v7_20261019/run_v7_20261019.py
"""
import os
assert os.environ.get('F1_TYRELIFE_2021') == 'v7'
HERE = os.path.dirname(os.path.abspath(__file__))
for name in ('us_outliers_20261014.py', 'table_tex_20261014.py'):
    path = os.path.join('results', 'outliers_us_20261014', name)
    src = open(path).read()
    old = "HERE = os.path.dirname(os.path.abspath(__file__))"
    assert src.count(old) == 1
    src = src.replace(old, f"HERE = {HERE!r}")
    exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
