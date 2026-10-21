"""2026-10-21 — Fig. 3 v7 con el eje y desde -1 (pedido del autor): fig3_v7_20261015.py con set_ylim(bottom=YMIN)
en los tres paneles. Salida: results/sim_v7_20261015/fig3_v7/ymin<YMIN>_laps<LAPS>/.
    F1_TYRELIFE_2021=v7 python results/sim_v7_20261015/fig3_v7_ymin_20261021.py --laps 28 [--ymin -1]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
YMIN = -1.0
if '--ymin' in sys.argv:
    i = sys.argv.index('--ymin'); YMIN = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
laps = sys.argv[sys.argv.index('--laps') + 1] if '--laps' in sys.argv else '56'
out = os.path.join(HERE, 'fig3_v7', f'ymin{YMIN:g}_laps{laps}')
os.makedirs(out, exist_ok=True)
src = open(os.path.join('results', 'fig3_20261014', 'plot_fig3_20261014.py')).read()
for a, b in [("'results/tyre_coef_study_20260930/pooled_hier_20261011/coefs_opcion3h.csv'",
              repr(os.path.join(HERE, 'tyre_v7', 'coefs_opcion3h.csv'))),
             ("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {out!r}"),
             ("ax.set_xlim(0, NLAPS); ax.grid(True, alpha=0.3)",
              f"ax.set_xlim(0, NLAPS); ax.set_ylim(bottom={YMIN!r}); ax.grid(True, alpha=0.3)")]:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, 'plot_fig3_20261014.py', 'exec'), {'__name__': '__main__'})
