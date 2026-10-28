import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
2026-10-27 — Chequeo de sensibilidad de la regla "bajo SC solo se para en la primera vuelta del periodo" (sc_bunching)
en el look-ahead (pedido del autor, chequeo (b)). Mismos argumentos que decision_support_rollouts.py; la decision debe
estar en la PRIMERA vuelta de un SC (Saudi 7, Miami 28, US 3).
    F1_SC2_BASE=belief   : envuelve ds_rollouts_belief_20261012.py (Figs. 8-10).
    F1_SC2_BASE=realhist : envuelve ds_rollouts_realhist_20261024.py (Fig. 11; requiere F1_DS_FORCE=hist).
    F1_SC_PIT_MAX_IDX=k  : en los rollouts el agente puede parar hasta la k-esima vuelta del periodo SC (1 = paper).
Ademas de las 4 acciones de siempre agrega, con k >= 2, los brazos 10+a = "no parar en n y parar en n+1 al compuesto a"
(despues decide la politica). Los rivales logit no cambian (siguen parando solo en la 1a vuelta). Con k = 1 y sin brazos
extra reproduce el CSV del barrido (control).
"""

import numpy as np

BASE = os.environ.get('F1_SC2_BASE', 'belief')
MAXIDX = int(os.environ.get('F1_SC_PIT_MAX_IDX', '1'))

if BASE == 'realhist':
    import ds_rollouts_realhist_20261024 as R
    D = R.D
    D.run_prefix_multi = R._prefix
else:
    import ds_rollouts_belief_20261012 as B
    D = B.D

from environment.f1_env_all_drivers import F1EnvAllDrivers

F1EnvAllDrivers.sc_pit_max_idx = MAXIDX
_HOOK = {'on': False, 'k': 0, 'act': 0}
_orig_step = F1EnvAllDrivers.step


def _step(self, action):
    if _HOOK['on']:
        _HOOK['k'] += 1
        if _HOOK['k'] == 2:   # la vuelta n+1 (el paso 1 es la accion forzada en n)
            action = _HOOK['act']
    return _orig_step(self, action)


F1EnvAllDrivers.step = _step

_orig_labels, _orig_rollout = D.action_labels, D.rollout_forced


def _labels(avail):
    lab = _orig_labels(avail)
    if MAXIDX >= 2:
        lab.update({10 + a: f'Stay, pit n+1 {lab[a]}' for a in range(1, len(avail) + 1)})
    return lab


def _rollout(model, vec_env, snap, obs_n, lstm, a, seed, **kw):
    if a < 10:
        return _orig_rollout(model, vec_env, snap, obs_n, lstm, a, seed, **kw)
    _HOOK.update(on=True, k=0, act=a - 10)
    try:
        pos, tt, pts, ns, pl, mh = _orig_rollout(model, vec_env, snap, obs_n, lstm, 0, seed, **kw)
    finally:
        _HOOK['on'] = False
    return pos, tt, pts, ns, pl, mh   # n_stops/pit_laps no cuentan la parada de n+1 (se ve en las medias)


D.action_labels, D.rollout_forced = _labels, _rollout

if __name__ == '__main__':
    print(f'[sc2] base={BASE} sc_pit_max_idx={MAXIDX}', flush=True)
    D.main()
