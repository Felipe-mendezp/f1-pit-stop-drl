"""Manifest de procedencia de los agentes multi-piloto v7 — 2026-10-15.

Mismo formato y logica que scripts/make_manifest_v6_20261014.py (se ejecuta con sus constantes cambiadas a v7, sin
modificarlo): tareas reproduce/cluster/tasks_v7.tsv, agentes trained_models/rl_agents_v7_20261015/selected,
simulador trained_models/simulation_v7_20261015.

    python scripts/make_manifest_v7_20261015.py            # -> reproduce/MODELS_MANIFEST_v7_20261015.csv
    python scripts/make_manifest_v7_20261015.py --check
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, 'scripts', 'make_manifest_v6_20261014.py')
src = open(path).read()
REPL = [("'tasks_v6.tsv'", "'tasks_v7.tsv'"),
        ("TREE = 'trained_models/rl_agents_v6_20261014'", "TREE = 'trained_models/rl_agents_v7_20261015'"),
        ("'MODELS_MANIFEST_v6_20261014.csv'", "'MODELS_MANIFEST_v7_20261015.csv'"),
        ("SIM = 'trained_models/simulation_v6_20261014'", "SIM = 'trained_models/simulation_v7_20261015'"),
        ("and v6 drivers (both set by the wrapper)", "and v6 drivers (both set by the wrapper); "
         "data/tyrelife_2021_factors_v7.json only for re-estimating the simulator"),
        ("t = tasks[f'v6__seed", "t = tasks[f'v7__seed"),
        ("notes=f'v6; selected", "notes=f'v7 (TyreLife 2021 x 0.716, K1 rival compound); selected")]
for a, b in REPL:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
sys.argv[0] = path
exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
