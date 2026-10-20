"""Mono-piloto (Fig. 5 / Tabla N.1) con la calibracion v5 de neumaticos (opcion 3, 2021-23) — 2026-10-11.

Mismo entrenamiento que el publicado (scripts/train_one_driver_paperhp_20260924.py: hiperparametros de la
tabla comentada F.1, normalizacion de la recompensa, 5M pasos, 10 entornos) y el mismo filtro de compuestos
< 5 % (scripts/one_driver_filter_20260925.py), con dos cambios:
  * simulador: trained_models/simulation_onedriver_opcion3_20261011/ (los GP enlazan a
    simulation_yfv4_opcion3_20261007; <GP>_fast_predictor.pkl enlaza a su laptime_clear.pkl, opcion 3
    2021-23). El piloto es VER (referencia de los dummies de piloto) y no hay SC/VSC, asi que la Ec. 2 y el
    modelo YF no intervienen.
  * filtro 5 %: recalculado sin las ediciones con > 10 % de vueltas WET/INT (filtro A de la opcion 3):
    data/compound_filter_one_driver_5pct_wetA_20261011.json.
Varias semillas: cada tarea entrena un (GP, algoritmo, semilla) en <agents-dir>/one_driver/<GP>/<algo>/.

    python scripts/one_driver_v5_20261011.py build                  # arbol + filtro (una vez)
    python scripts/one_driver_v5_20261011.py verify                 # QMIP filtrado == DP exacto (8 GP)
    python scripts/one_driver_v5_20261011.py train --agents-dir <dir> <args de train_one_driver_paperhp>
    python scripts/one_driver_v5_20261011.py screen --agents-dir <dir> <args de screen_one_driver_20260923>
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = 'trained_models/simulation_onedriver_opcion3_20261011'
BASE = 'simulation_yfv4_opcion3_20261007'
GPS8 = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
        'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']

# --agents-dir y el simulador se fijan ANTES de importar config
if '--agents-dir' in sys.argv:
    i = sys.argv.index('--agents-dir')
    os.environ['F1_RL_AGENTS_DIR'] = os.path.join(_ROOT, sys.argv[i + 1]) if not os.path.isabs(sys.argv[i + 1]) else sys.argv[i + 1]
    del sys.argv[i:i + 2]
os.environ.setdefault('F1_SIMULATION_DIR', os.path.join(_ROOT, TREE))
os.environ.setdefault('F1_ONE_DRIVER_DSQ_PENALTY', '-1000')

sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
import config
import compat; compat.setup_all()
import one_driver_filter_20260925 as F

FILTER = os.path.join(str(config.DATA_DIR), 'compound_filter_one_driver_5pct_wetA_20261011.json')


def build():
    tree = os.path.join(_ROOT, TREE)
    os.makedirs(tree, exist_ok=True)
    for gp in GPS8:
        assert os.path.isfile(os.path.join(_ROOT, 'trained_models', BASE, gp, 'laptime_clear.pkl')), gp
        for name, target in [(gp, os.path.join('..', BASE, gp)),
                             (f'{gp}_fast_predictor.pkl', os.path.join(gp, 'laptime_clear.pkl'))]:
            p = os.path.join(tree, name)
            if not os.path.lexists(p):
                os.symlink(target, p)
    with open(os.path.join(tree, 'README.md'), 'w') as f:
        f.write('Mono-piloto v5 (2026-10-11, scripts/one_driver_v5_20261011.py build): los GP enlazan a '
                f'{BASE} (opcion 3 2021-23); <GP>_fast_predictor.pkl = <GP>/laptime_clear.pkl.\n')
    # filtro 5 % sin ediciones humedas (filtro A)
    import importlib.util
    import models.regressions as R
    import plot_fig5_stints_one_driver as P
    spec = importlib.util.spec_from_file_location(
        'pooled', os.path.join(_ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_20261004.py'))
    pooled = importlib.util.module_from_spec(spec); spec.loader.exec_module(pooled)
    hist = pooled.hist_laps_A()          # usa el R.all_laps original
    R.all_laps = lambda outliers=False: hist.copy()
    P.SELECTED_GP = {gp: c for gp, c in P.SELECTED_GP.items() if gp in GPS8}
    F.build_filter(out=FILTER)
    print('arbol:', tree)


def verify():
    """Igual que one_driver_filter_20260925.verify, sobre los 8 GP."""
    import plot_fig5_stints_one_driver as P
    P.SELECTED_GP = {gp: c for gp, c in P.SELECTED_GP.items() if gp in GPS8}
    F.verify()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build':
        return build()
    assert os.path.isfile(FILTER), 'correr primero: build'
    F.install(FILTER)
    if cmd == 'verify':
        return verify()
    if cmd in ('train', 'screen', 'fig5', 'm1'):
        print('agents:', config.RL_AGENTS_DIR, '| sim:', os.environ['F1_SIMULATION_DIR'], flush=True)
        return F.main()
    print(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
