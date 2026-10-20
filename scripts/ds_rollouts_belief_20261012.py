import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-12 — Apoyo a la decision "creencia / mundo": la carrera realizada (prefijo comun hasta la
vuelta n) se simula en el mundo 2024 (neumaticos 2021-24 de F1_WORLD2024_TYRES: opcion3 u opcion3h; igual que
scripts/ds_rollouts_world2024_20261011.py), pero las proyecciones desde la vuelta n (rollouts de
cada accion candidata) usan los neumaticos que el decisor puede conocer antes de la carrera
(2021-23 = simulador cargado). El agente y su continuacion no cambian (entrenados con 2021-23).
Mismos argumentos que decision_support_rollouts.py:
    python scripts/ds_rollouts_belief_20261012.py --gp 'United States Grand Prix' ...
"""

import ds_rollouts_world2024_20261011  # noqa: F401  (parcha ModelLoader.load: mundo 2024 al cargar)
import evaluate_world2024_20261010 as W
import decision_support_rollouts as D

_orig_prefix, _orig_rollout = D.run_prefix_multi, D.rollout_forced


def _set(which):
    for s in W._SPEC.values():
        s['pred'].coef_[:] = s[which]


def _prefix(*a, **kw):
    _set('world')
    out = _orig_prefix(*a, **kw)
    _set('base')
    return out


def _rollout(*a, **kw):
    _set('base')
    return _orig_rollout(*a, **kw)


D.run_prefix_multi, D.rollout_forced = _prefix, _rollout

if __name__ == '__main__':
    D.main()
