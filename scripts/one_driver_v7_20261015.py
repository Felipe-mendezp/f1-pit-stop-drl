"""Mono-piloto v7 (simulador trained_models/simulation_onedriver_v7_20261015, sin filtro de compuestos) — 2026-10-15.

Igual que scripts/one_driver_v5hnf_20261014.py (sin filtro: data/compound_filter_one_driver_none_20261014.json,
hiperparametros publicados), pero con el arbol mono-piloto v7 (los 8 GP -> simulation_v7_20261015). Para screen, fig5
y m1 se ejecuta scripts/one_driver_v5h_20261013.py con su arbol fijo reemplazado por el de v7 (sin modificar el
archivo).

    python scripts/one_driver_v7_20261015.py verify         # QMIP == DP exacto (8 GP, 3 compuestos)
    python scripts/one_driver_v7_20261015.py train  --agents-dir <dir> <args de train_one_driver_paperhp>
    python scripts/one_driver_v7_20261015.py screen --agents-dir <dir> <args de screen_one_driver_20260923>
    python scripts/one_driver_v7_20261015.py fig5   --agents-dir <dir> --models-dir <dir>/one_driver --output <png>
    python scripts/one_driver_v7_20261015.py m1     --agents-dir <dir> --out-dir <dir>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_v7_20261015')
OLD_TREE = "os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_opcion3h8_20261012')"
NONE = os.path.join(_ROOT, 'data', 'compound_filter_one_driver_none_20261014.json')

if '--agents-dir' in sys.argv:            # antes de importar config
    i = sys.argv.index('--agents-dir')
    p = sys.argv[i + 1]
    os.environ['F1_RL_AGENTS_DIR'] = p if os.path.isabs(p) else os.path.join(_ROOT, p)
    del sys.argv[i:i + 2]
os.environ['F1_SIMULATION_DIR'] = TREE
os.environ.setdefault('F1_ONE_DRIVER_DSQ_PENALTY', '-1000')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    assert os.path.isfile(NONE), NONE
    assert os.path.isdir(TREE), TREE
    import config  # noqa: F401  (fija rutas con las variables de entorno de arriba)
    import compat; compat.setup_all()
    import one_driver_filter_20260925 as F
    _install = F.install
    F.install = lambda path=None: _install(NONE)
    print(f'[v7 sin filtro] {cmd} | sim: {TREE} | filtro: {NONE}', flush=True)
    if cmd in ('train', 'verify'):
        sys.argv[0] = os.path.join(_ROOT, 'scripts', 'one_driver_v5_20261011.py')
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif cmd in ('screen', 'fig5', 'm1'):
        path = os.path.join(_ROOT, 'scripts', 'one_driver_v5h_20261013.py')
        src = open(path).read()
        assert src.count(OLD_TREE) == 1, 'cambio one_driver_v5h_20261013.py'
        src = src.replace(OLD_TREE, repr(TREE))
        sys.argv[0] = path
        exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
    else:
        print(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
