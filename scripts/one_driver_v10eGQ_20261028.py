"""Mono-piloto v10eGQ (Fig. 5 / Tabla N.1) — 2026-10-28, version adoptada por el autor: edad efectiva GQ (sin vueltas
SC/VSC en carrera; edad de los juegos usados en vueltas de empuje de practicas, sprint y clasificacion;
results/tyrelife_effective_20261028/README.md). Igual que scripts/one_driver_v10d_20261028.py con el arbol
trained_models/simulation_onedriver_v10eGQ_20261028 (via scripts/one_driver_v10e_20261028.py con F1_V10E=GQ). Las edades
de los juegos 2024 en vueltas de empuje las fija scripts/tyre_ages_v10eGQ_20261028.py, que va primero en las tareas.
    python scripts/one_driver_v10eGQ_20261028.py {verify|train|screen|fig5|m1} <args de one_driver_v7_20261015.py>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['F1_V10E'] = 'GQ'
if __name__ == '__main__':
    target = os.path.join(_ROOT, 'scripts', 'one_driver_v10e_20261028.py')
    sys.argv = [target] + sys.argv[1:]
    runpy.run_path(target, run_name='__main__')
