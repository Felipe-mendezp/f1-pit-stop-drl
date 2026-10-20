"""Mono-piloto v5h SIN filtro de compuestos (homologado con el multi-piloto) — 2026-10-14.

Decision del autor (2026-10-14): se elimina el filtro de compuestos < 5 % del mono-piloto. Los compuestos
poco usados se calibran con el modelo conjunto GLS (opcion3h), igual que en el multi-piloto, que nunca tuvo
filtro. Todo lo demas es igual al mono-piloto v5h (simulador trained_models/simulation_onedriver_opcion3h8_20261012,
hiperparametros publicados, 8 GP, semillas 1-5).

Implementacion: se reutilizan los scripts existentes, pero one_driver_filter_20260925.install() se redirige a
data/compound_filter_one_driver_none_20261014.json (keep = los 3 compuestos de 2024 en cada GP, dropped = []).
Con ese JSON el entorno filtrado queda con los 3 compuestos y Discrete(4), igual que el entorno original, y el
QMIP no penaliza ningun compuesto.

    python scripts/one_driver_v5hnf_20261014.py build-none     # escribe el JSON sin filtro (una vez)
    python scripts/one_driver_v5hnf_20261014.py verify         # QMIP == DP exacto (8 GP, 3 compuestos)
    python scripts/one_driver_v5hnf_20261014.py train  --agents-dir <dir> <args de train_one_driver_paperhp>
    python scripts/one_driver_v5hnf_20261014.py screen --agents-dir <dir> <args de screen_one_driver_20260923>
    python scripts/one_driver_v5hnf_20261014.py fig5   --agents-dir <dir> --models-dir <dir>/one_driver --output <png>
    python scripts/one_driver_v5hnf_20261014.py m1     --agents-dir <dir> --out-dir <dir>
"""
import os, sys, json, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_opcion3h8_20261012')
WETA = os.path.join(_ROOT, 'data', 'compound_filter_one_driver_5pct_wetA_20261011.json')
NONE = os.path.join(_ROOT, 'data', 'compound_filter_one_driver_none_20261014.json')

# --agents-dir y el simulador se fijan ANTES de importar config
if '--agents-dir' in sys.argv:
    i = sys.argv.index('--agents-dir')
    p = sys.argv[i + 1]
    os.environ['F1_RL_AGENTS_DIR'] = p if os.path.isabs(p) else os.path.join(_ROOT, p)
    del sys.argv[i:i + 2]
os.environ.setdefault('F1_SIMULATION_DIR', TREE)
os.environ.setdefault('F1_ONE_DRIVER_DSQ_PENALTY', '-1000')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))


def build_none():
    with open(WETA) as f:
        d = json.load(f)
    for v in d['gps'].values():
        v['keep'] = list(v['compounds_2024']); v['dropped'] = []
    d['threshold'] = 0.0
    d['note'] = ('2026-10-14: sin filtro (homologado con el multi-piloto); share_2021_2023 copiado de '
                 'compound_filter_one_driver_5pct_wetA_20261011.json solo como referencia.')
    with open(NONE, 'w') as f:
        json.dump(d, f, indent=2)
    print('->', NONE)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'build-none':
        return build_none()
    assert os.path.isfile(NONE), 'correr primero: build-none'
    assert os.path.realpath(os.environ['F1_SIMULATION_DIR']) == os.path.realpath(TREE), os.environ['F1_SIMULATION_DIR']
    import config  # noqa: F401  (fija rutas con las variables de entorno de arriba)
    import compat; compat.setup_all()
    import one_driver_filter_20260925 as F
    _install = F.install
    F.install = lambda path=None: _install(NONE)
    if cmd in ('train', 'verify'):
        target = 'one_driver_v5_20261011.py'
    elif cmd in ('screen', 'fig5', 'm1'):
        target = 'one_driver_v5h_20261013.py'      # 8 GP (sin Emilia Romagna) y 8 paneles; mismo arbol opcion3h8
    else:
        print(__doc__); sys.exit(2)
    print(f'[sin filtro] {cmd} via {target} | filtro: {NONE}', flush=True)
    sys.argv[0] = os.path.join(_ROOT, 'scripts', target)
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
