"""2026-10-15 — Diagnostico TRPO Miami (Fig. 7 v6): traza vuelta a vuelta de la politica.

Corre scripts/evaluate_world2024_20261010.py --fig 7 (mismo protocolo, mundo 2024 opcion3h, agentes v6 seleccionados)
y registra en cada paso: vuelta, SC/VSC, vueltas restantes de SC/VSC visibles (F1_SC_REVEAL), compuesto y vida del
neumatico del agente, posicion, accion elegida y probabilidad de cada accion segun la politica (TRPO/A2C/PPO).
    F1_SIMULATION_DIR=trained_models/simulation_v6_20261014 F1_RL_AGENTS_DIR=trained_models/rl_agents_v6_20261014/selected \
    F1_WORLD2024_TYRES=opcion3h python results/world2024_v6_20261014/miami_trpo/trace_20261015.py --algos TRPO A2C --n-sims 20
Salida: results/world2024_v6_20261014/miami_trpo/trace_<algos>.csv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd, torch
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers
from stable_baselines3.common.base_class import BaseAlgorithm

ROWS, CUR = [], {}
_pred = BaseAlgorithm.predict
def _predict(self, obs, state=None, episode_start=None, deterministic=False):
    try:
        with torch.no_grad():
            o, _ = self.policy.obs_to_tensor(obs)
            d = self.policy.get_distribution(o) if hasattr(self.policy, 'get_distribution') else None
            CUR['probs'] = d.distribution.probs.cpu().numpy()[0] if d is not None else None
    except Exception:
        CUR['probs'] = None
    CUR['algo'] = type(self).__name__
    return _pred(self, obs, state=state, episode_start=episode_start, deterministic=deterministic)
BaseAlgorithm.predict = _predict

_step = F1EnvAllDrivers.step
def _s(self, action):
    a = self.agent_driver
    row = dict(algo=CUR.get('algo'), lap=int(self.lap_number), sc=int(self.sc), vsc=int(self.vsc),
               compound=int(a.compound), tyrelife=float(a.tyrelife), pos=int(self.agent_position),
               changed=bool(a.has_changed_compound), action=int(np.asarray(action).ravel()[0]))
    p = CUR.get('probs')
    if p is not None:
        row.update({f'p{i}': float(x) for i, x in enumerate(p)})
    ROWS.append(row)
    return _step(self, action)
F1EnvAllDrivers.step = _s

# NOYF=1: misma Fig. 7 pero sin SC/VSC (control: aisla el efecto de las neutralizaciones reales de 2024)
NOYF = os.environ.get('NOYF') == '1'
if NOYF:
    _lgc = W.E7.load_gp_config
    def _no_yf(gp):
        c = dict(_lgc(gp)); c['laps_sc_2024'] = []; c['laps_vsc_2024'] = []; return c
    W.E7.load_gp_config = _no_yf


if __name__ == '__main__':
    args = sys.argv[1:]
    out = os.path.join(HERE, 'fig7_run' + ('_noyf' if NOYF else ''))
    algos = args[args.index('--algos') + 1:] if '--algos' in args else ['TRPO']
    algos = [x for x in algos if not x.startswith('--')][:5]
    sys.argv = ['evaluate_world2024_20261010.py', '--fig', '7', '--world', '2024', '--gps', 'Miami_OCO', '--out', out] + args
    W.main()
    D = pd.DataFrame(ROWS)
    D['race'] = (D.lap.diff() < 0).cumsum()
    f = os.path.join(HERE, f"trace_{'_'.join(a for a in algos if not a.isdigit())}{'_noyf' if NOYF else ''}.csv")
    D.to_csv(f, index=False)
    print('->', f, len(D))
