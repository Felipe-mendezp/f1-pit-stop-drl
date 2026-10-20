"""2026-10-15 — Fig. 3 con la calibracion v7: mismo script que results/fig3_20261014/plot_fig3_20261014.py (sin
modificarlo), con la vida del neumatico de 2021 x 0.716 (correr con F1_TYRELIFE_2021=v7) y el modelo conjunto de
tyre_v7/coefs_opcion3h.csv. Salida: results/sim_v7_20261015/fig3_v7/.
    F1_TYRELIFE_2021=v7 python results/sim_v7_20261015/fig3_v7_20261015.py [--gp ... --laps ...]
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join('results', 'fig3_20261014', 'plot_fig3_20261014.py')).read()
src = src.replace("'results/tyre_coef_study_20260930/pooled_hier_20261011/coefs_opcion3h.csv'",
                  repr(os.path.join(HERE, 'tyre_v7', 'coefs_opcion3h.csv')))
src = src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {os.path.join(HERE, 'fig3_v7')!r}")
assert 'tyre_v7' in src and 'fig3_v7' in src
exec(compile(src, 'plot_fig3_20261014.py', 'exec'), {'__name__': '__main__'})
