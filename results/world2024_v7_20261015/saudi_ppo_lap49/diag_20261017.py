"""Diagnostico (2026-10-17): parada en la vuelta 49 de 50 de PPO (semilla 5) en Saudi, Fig. 7 v7 (31 % de las carreras).
Envuelve scripts/evaluate_world2024_20261010.py (misma evaluacion y semillas, parcheada en memoria) y registra, para
cada paso del agente DRL desde la vuelta 40: accion, posicion, compuesto, TyreLife, distancia en s al rival de adelante
y al de atras (sin rivales retirados), SC/VSC y rivales retirados.
    --mode trace    sin cambios en la simulacion (debe reproducir la Fig. 7)
    --mode nolate   contrafactual: el agente DRL no puede parar desde la vuelta --from-lap (accion forzada a 0)
    ... --mode <m> [--from-lap 45] <args de evaluate_world2024_20261010.py>
Salida: <out>/fig7_2024/<GP>.csv (como la Fig. 7) y <out>/trace_<mode>.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
i = sys.argv.index('--mode'); MODE = sys.argv[i + 1]; del sys.argv[i:i + 2]
FROM = 45
if '--from-lap' in sys.argv:
    i = sys.argv.index('--from-lap'); FROM = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
assert MODE in ('trace', 'nolate'), MODE
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

_reset = F1EnvAllDrivers.reset
def reset(self, *a, **k):
    ST['race'] += 1; ST['drl'] = False
    return _reset(self, *a, **k)
F1EnvAllDrivers.reset = reset

_step = F1EnvAllDrivers.step   # ya parcheado por W (registro de paradas): se envuelve encima
def step(self, action):
    drl, ST['drl'] = ST['drl'], False
    lap = int(self.lap_number)
    a = int(np.asarray(action).ravel()[0]) if np.ndim(action) else int(action)
    if drl and lap >= 40:
        ag = self.agent_driver
        td = [d.total_time - ag.total_time for d in self.rival_drivers if not d.is_dnf]
        ahead = [-x for x in td if x < 0]; behind = [x for x in td if x > 0]
        ROWS.append(dict(race=ST['race'], lap=lap, action=a, position=ag.position, compound=ag.compound,
                         tyrelife=ag.tyrelife, gap_ahead=min(ahead) if ahead else np.nan,
                         gap_behind=min(behind) if behind else np.nan, sc=int(self.sc > 0), vsc=int(self.vsc > 0),
                         n_dnf=sum(d.is_dnf for d in self.rival_drivers)))
    if drl and MODE == 'nolate' and lap >= FROM and a != 0:
        action = 0
    out = _step(self, action)
    if drl and ROWS and ROWS[-1]['race'] == ST['race'] and ROWS[-1]['lap'] == lap:
        ROWS[-1]['position_after'] = self.agent_driver.position
    return out
F1EnvAllDrivers.step = step

print('diagnostico:', MODE, 'desde vuelta', FROM, flush=True)
W.main()
pd.DataFrame(ROWS).to_csv(os.path.join(OUT, f'trace_{MODE}.csv'), index=False)
