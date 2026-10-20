import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-11 — Apoyo a la decision en el "mundo 2024": decision_support_rollouts.py con el entorno
simulando con los coeficientes de neumatico opcion3 re-estimados con 2021-2024, mientras el agente
(entrenado en F1_SIMULATION_DIR, opcion3 2021-23) decide con lo que aprendio. Mismo cambio de
coeficientes que scripts/evaluate_world2024_20261010.py (register: solo offset por compuesto y
TyreLife x compuesto; controla que el simulador cargado tenga opcion3 2021-23).
Mismos argumentos que decision_support_rollouts.py:
    python scripts/ds_rollouts_world2024_20261011.py --gp 'Hungarian Grand Prix' ... 
"""

import runpy
import evaluate_world2024_20261010 as W
from model_loader import ModelLoader

_orig_load = ModelLoader.load
def _load(self, gp, use_cache=True):
    m = _orig_load(self, gp, use_cache)
    if gp not in W._SPEC or W._SPEC[gp]['pred'] is not m.laptime_clear:
        changed = W.register(gp, m.laptime_clear, '2024')
        print(f'MUNDO 2024 {gp}: {len(changed)} coeficientes de neumatico cambiados', flush=True)
    m.laptime_clear.coef_[:] = W._SPEC[gp]['world']
    return m
ModelLoader.load = _load

if __name__ == '__main__':
    sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_support_rollouts.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
