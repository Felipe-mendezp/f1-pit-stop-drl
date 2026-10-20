"""Probabilidades de accion de RPPO vuelta a vuelta en la Fig. 7 v5h (mundo 2024) — 2026-10-14 (diagnostico).

Repite scripts/evaluate_world2024_20261010.py (mismas semillas) y, en cada decision de RecurrentPPO, guarda la
distribucion de la politica (softmax sobre las 4 acciones: 0 = seguir, 1..3 = parar a cada compuesto) y el estado
del agente (vuelta, posicion, compuesto, edad del neumatico). Parchea en memoria; no cambia la evaluacion.
    ... trace en <json>: una lista por carrera de dicts.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
i = sys.argv.index('--trace-out'); OUT = sys.argv[i + 1]; del sys.argv[i:i + 2]
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

RACES, CUR, ENV = [], [], {}
_reset, _step = F1EnvAllDrivers.reset, F1EnvAllDrivers.step


def reset(self, *a, **k):
    global CUR
    if CUR:
        RACES.append(CUR)
    CUR = []; ENV['e'] = self
    return _reset(self, *a, **k)


def step(self, action):
    if CUR and CUR[-1].get('action') is None:
        CUR[-1]['action'] = int(action)
    return _step(self, action)


_pred = RecurrentActorCriticPolicy._predict


def predict(self, observation, lstm_states, episode_starts, deterministic=False):
    dist, _ = self.get_distribution(observation, lstm_states, episode_starts)
    p = dist.distribution.probs.detach().cpu().numpy()[0].round(4).tolist()
    e = ENV['e']; a = e.agent_driver
    CUR.append(dict(lap=int(e.lap_number), pos=int(e.agent_position), comp=int(a.compound),
                    life=float(a.tyrelife), probs=p, action=None))
    return _pred(self, observation, lstm_states, episode_starts, deterministic)


F1EnvAllDrivers.reset, F1EnvAllDrivers.step = reset, step
RecurrentActorCriticPolicy._predict = predict
try:
    W.main()
finally:
    if CUR:
        RACES.append(CUR)
    RACES[:] = [r for r in RACES if any(d.get('probs') for d in r)]
    json.dump(RACES, open(OUT, 'w'))
    print('carreras RPPO:', len(RACES), '->', OUT)
