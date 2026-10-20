"""2026-10-19 — Fig. 3 v7 con eje y desde --ymin (pedido del autor: -1) y eje x hasta --laps (32). Ejecuta
results/sim_v7_20261015/fig3_v7_20261015.py (que a su vez ejecuta plot_fig3_20261014.py sin modificarlo) agregando
ax.set_ylim(bottom=YMIN) junto a set_xlim. Salida: results/sim_v7_20261015/fig3_v7/.
    F1_TYRELIFE_2021=v7 python results/sim_v7_20261015/fig3_v7_ylim_20261019.py --laps 32 --ymin -1
"""
import sys, builtins
i = sys.argv.index('--ymin'); YMIN = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
_open = builtins.open
OLD = 'ax.set_xlim(0, NLAPS); ax.grid(True, alpha=0.3)'


def _patched_open(path, *a, **k):
    f = _open(path, *a, **k)
    if str(path).endswith('plot_fig3_20261014.py'):
        src = f.read(); f.close()
        assert src.count(OLD) == 1
        import io
        return io.StringIO(src.replace(OLD, OLD + f'; ax.set_ylim(bottom={YMIN})'))
    return f


builtins.open = _patched_open
exec(compile(_open('results/sim_v7_20261015/fig3_v7_20261015.py').read(), 'fig3_v7_20261015.py', 'exec'),
     {'__name__': '__main__', '__file__': 'results/sim_v7_20261015/fig3_v7_20261015.py'})
