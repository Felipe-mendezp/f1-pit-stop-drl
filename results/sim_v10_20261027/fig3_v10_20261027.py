"""2026-10-27 — Fig. 3 con la calibracion v10: mismo script que results/fig3_20261014/plot_fig3_20261014.py (sin
modificarlo), sin las vueltas de paso por el pit lane (common_v10_20261027.patch_regressions), vida del neumatico de 2021
x 0.740 (F1_TYRELIFE_2021=v10), el modelo conjunto de tyre_v10/coefs_opcion3h.csv y el eje y desde -1 (como la version
del paper, results/sim_v7_20261015/fig3_v7_ymin_20261021.py). Salida: results/sim_v10_20261027/fig3_v10/.
    F1_TYRELIFE_2021=v10 python results/sim_v10_20261027/fig3_v10_20261027.py --laps 28
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'src'), HERE]
assert os.environ.get('F1_TYRELIFE_2021') == 'v10'
import config, compat; compat.setup_all()
import models.regressions as R
import common_v10_20261027 as CV
CV.patch_regressions(R)
src = open(os.path.join('results', 'fig3_20261014', 'plot_fig3_20261014.py')).read()
src = src.replace("'results/tyre_coef_study_20260930/pooled_hier_20261011/coefs_opcion3h.csv'",
                  repr(os.path.join(HERE, 'tyre_v10', 'coefs_opcion3h.csv')))
src = src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {os.path.join(HERE, 'fig3_v10')!r}")
a = "ax.set_xlim(0, NLAPS); ax.grid(True, alpha=0.3)"   # eje y desde -1, como la version del paper (fig3_v7_ymin_20261021.py)
assert src.count(a) == 1
src = src.replace(a, "ax.set_xlim(0, NLAPS); ax.set_ylim(bottom=-1.0); ax.grid(True, alpha=0.3)")
assert 'tyre_v10' in src and 'fig3_v10' in src
exec(compile(src, 'plot_fig3_20261014.py', 'exec'), {'__name__': '__main__'})
