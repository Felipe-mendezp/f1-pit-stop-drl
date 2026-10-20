"""Copia de ../bahrain_trpo_dsq/diag_20261018.py que tambien intercepta DQN.predict (DQN no usa BaseAlgorithm.predict).
Diagnostico (2026-10-18): TRPO Bahrain (semilla 5, reglas 5/7/8) descalificado en el 97 % de la Fig. 7 v7.
Envuelve scripts/evaluate_world2024_20261010.py sin cambiar la simulacion y registra, en cada paso del agente DRL
(desde la vuelta 0 = eleccion del compuesto de partida): accion, compuesto montado antes y despues, TyreLife,
regla de dos compuestos cumplida, posicion y SC/VSC.
    ... diag_20261018.py <args de evaluate_world2024_20261010.py>
Salida: <out>/fig<6|7>_2024[_realdnf]/<GP>.csv (como la Fig. 6/7) y <out>/trace.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = sys.argv[sys.argv.index('--out') + 1]
import numpy as np, pandas as pd
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers
from stable_baselines3.common.base_class import BaseAlgorithm

ROWS, ST = [], dict(drl=False, race=-1)
_pred = BaseAlgorithm.predict
def predict(self, *a, **k):
    ST['drl'] = True            # el siguiente step es de un agente DRL (la QMIP no llama predict)
    return _pred(self, *a, **k)
BaseAlgorithm.predict = predict
from stable_baselines3 import DQN
_pred_dqn = DQN.predict
def predict_dqn(self, *a, **k):
    ST['drl'] = True
    return _pred_dqn(self, *a, **k)
DQN.predict = predict_dqn

_reset = F1EnvAllDrivers.reset
def reset(self, *a, **k):
    ST['race'] += 1; ST['drl'] = False
    return _reset(self, *a, **k)
F1EnvAllDrivers.reset = reset

_step = F1EnvAllDrivers.step
def step(self, action):
    drl, ST['drl'] = ST['drl'], False
    if not drl:
        return _step(self, action)
    ag = self.agent_driver
    a = int(np.asarray(action).ravel()[0]) if np.ndim(action) else int(action)
    row = dict(race=ST['race'], lap=int(self.lap_number), n_laps=self.n_laps, action=a,
               comp_before=int(ag.compound or 0), tyrelife=int(ag.tyrelife or 0),
               changed_before=int(ag.has_changed_compound), position=ag.position,
               sc=int(self.sc > 0), vsc=int(self.vsc > 0), avail=' '.join(map(str, self.available_compounds)))
    out = _step(self, action)
    row.update(comp_after=int(ag.compound or 0), changed_after=int(ag.has_changed_compound))
    if row['lap'] <= 1:   # 2026-10-18: compuestos de los rivales (en la v0 aun no eligen; en la v1 ya)
        rc = [int(d.compound or 0) for d in self.rival_drivers]
        row.update({f'riv_c{c}': rc.count(c) for c in [0] + list(self.available_compounds)})
    ROWS.append(row)
    return out
F1EnvAllDrivers.step = step

W.main()
pd.DataFrame(ROWS).to_csv(os.path.join(OUT, 'trace.csv'), index=False)
