"""Diagnostico 2026-10-14: para cada parada del agente registra si esa vuelta estaba bajo SC o VSC (env.sc / env.vsc
despues del step, que reflejan la vuelta simulada). Misma evaluacion que scripts/evaluate_world2024_20261010.py.
    ... --trace-out <json> <args de evaluate_world2024_20261010.py>
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
i = sys.argv.index('--trace-out'); OUT = sys.argv[i + 1]; del sys.argv[i:i + 2]
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers
RACES, CUR = [], []
_reset, _step = F1EnvAllDrivers.reset, F1EnvAllDrivers.step


def reset(self, *a, **k):
    global CUR
    if CUR:
        RACES.append(CUR)
    CUR = []
    return _reset(self, *a, **k)


def step(self, action):
    lap = int(self.lap_number)
    out = _step(self, action)
    if lap >= 1 and int(action) != 0:
        CUR.append((lap, int(bool(self.sc)), int(bool(self.vsc))))
    elif lap == 0:
        CUR.append(('start',))
    return out


F1EnvAllDrivers.reset, F1EnvAllDrivers.step = reset, step
try:
    W.main()
finally:
    if CUR:
        RACES.append(CUR)
    json.dump(RACES, open(OUT, 'w')); print('carreras', len(RACES))
