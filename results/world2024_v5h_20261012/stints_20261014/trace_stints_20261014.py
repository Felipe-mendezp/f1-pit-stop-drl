"""Stints por carrera en la Fig. 6/7 v5h (mundo 2024) — 2026-10-14.

Repite scripts/evaluate_world2024_20261010.py (mismas semillas, mismos agentes seleccionados, mismo mundo opcion3h)
registrando vuelta a vuelta el compuesto del agente, la edad del neumatico y la accion
(una parada al mismo compuesto no cambia el compuesto: se detecta porque la edad vuelve a 0). Parchea F1EnvAllDrivers.reset/step en memoria; no cambia la
evaluacion. Las carreras salen en el orden de evaluacion: --algos en orden y luego QMIP, n_sims cada uno.
Control: la posicion final de cada carrera debe coincidir con fig<6|7>_2024/<GP>.csv (se verifica en summarize).

    source reproduce/env_paper_v3_20261007.sh
    F1_WORLD2024_TYRES=opcion3h F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012 \
    F1_RL_AGENTS_DIR=trained_models/rl_agents_v5h_20261011/selected \
    python results/world2024_v5h_20261012/stints_20261014/trace_stints_20261014.py --fig 7 --gps Miami_OCO \
        --algos RecurrentPPO PPO --n-sims 500 --trace-out <csv> --out <dir>
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
i = sys.argv.index('--trace-out'); TRACE_OUT = sys.argv[i + 1]; del sys.argv[i:i + 2]

import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers

TRACES, CUR = [], []
_reset, _step = F1EnvAllDrivers.reset, F1EnvAllDrivers.step


def reset(self, *a, **k):
    global CUR
    if CUR:
        TRACES.append(CUR)
    CUR = []
    return _reset(self, *a, **k)


def step(self, action):
    lap = int(self.lap_number)
    out = _step(self, action)
    CUR.append((lap, int(self.agent_driver.compound), float(self.agent_driver.tyrelife), int(action)))
    return out


F1EnvAllDrivers.reset, F1EnvAllDrivers.step = reset, step
try:
    W.main()
finally:
    if CUR:
        TRACES.append(CUR)
    with open(TRACE_OUT, 'w') as f:
        json.dump(TRACES, f)
    print(f'trazas: {len(TRACES)} -> {TRACE_OUT}')
