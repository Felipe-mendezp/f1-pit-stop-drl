"""Mono-piloto v5h (opcion3h 2021-23) — envoltorio de scripts/one_driver_v5_20261011.py — 2026-10-13.

Igual que one_driver_v5_20261011.py pero fija el simulador al arbol con el que se entrenaron los agentes
trained_models/rl_agents_onedriver_v5h_20261011/ (trained_models/simulation_onedriver_opcion3h8_20261012/;
el predictor de Saudi es identico al de simulation_onedriver_opcion3h_20261011, usado por los 25 jobs f1odh).
Filtro 5 %: el mismo de one_driver_v5 (compound_filter_one_driver_5pct_wetA_20261011.json).

    python scripts/one_driver_v5h_20261013.py screen --agents-dir <dir> <args de screen_one_driver_20260923>
    python scripts/one_driver_v5h_20261013.py fig5 --models-dir <dir>/one_driver --output <png>
    python scripts/one_driver_v5h_20261013.py m1 --agents-dir <dir> --out-dir <dir>      # Tabla N.1
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['F1_SIMULATION_DIR'] = os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_opcion3h8_20261012')
if '--agents-dir' in sys.argv:            # igual que one_driver_v5, antes de importar config
    i = sys.argv.index('--agents-dir')
    os.environ['F1_RL_AGENTS_DIR'] = os.path.join(_ROOT, sys.argv[i + 1]) if not os.path.isabs(sys.argv[i + 1]) else sys.argv[i + 1]
    del sys.argv[i:i + 2]
# screen_one_driver_20260923 recorre todo SELECTED_GP (incluye GP fuera de los 8, sin predictor en el arbol):
# se restringe en el mismo dict (in place) a los 8 GP, como hace verify() en one_driver_v5.
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
GPS8 = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
        'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
os.environ.setdefault('F1_ONE_DRIVER_DSQ_PENALTY', '-1000')      # igual que one_driver_v5
import config
import compat; compat.setup_all()
import one_driver_filter_20260925 as F
# el filtro 5 % debe instalarse ANTES de importar plot_fig5_stints_one_driver (que copia el entorno y la QMIP
# al importarse); one_driver_v5 vuelve a llamar F.install, que es idempotente.
F.install(os.path.join(str(config.DATA_DIR), 'compound_filter_one_driver_5pct_wetA_20261011.json'))
import plot_fig5_stints_one_driver as P
for _gp in [g for g in P.SELECTED_GP if g not in GPS8]:
    del P.SELECTED_GP[_gp]
# fig5: plot_fig5_stints_one_driver dibuja una grilla 3x3 que asume 9 GP; con 8 se oculta el ultimo panel.
_subplots = P.plt.subplots
def _subplots8(*a, **k):
    fig, axs = _subplots(*a, **k)
    flat = list(axs.flat)
    for ax in flat[len(P.SELECTED_GP):]:
        ax.set_visible(False)
    class _Axs:
        pass
    out = _Axs(); out.flat = flat[:len(P.SELECTED_GP)]
    return fig, out
P.plt.subplots = _subplots8
sys.argv[0] = os.path.join(_ROOT, 'scripts', 'one_driver_v5_20261011.py')
runpy.run_path(sys.argv[0], run_name='__main__')
