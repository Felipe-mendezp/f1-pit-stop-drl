"""Diagnostico (2026-10-14): ¿que gatilla las paradas sin sentido de los agentes DRL en la Fig. 7 v5h (mundo 2024)?
Hipotesis: los abandonos (DNF) de rivales. Solo existen en la Fig. 7 (estrategias reales, -1 = abandono); en el
entrenamiento y en la Fig. 6 los rivales nunca abandonan, y un rival retirado aparece en la observacion con
time_diff_to_agent = +500 (cota del espacio), un valor que el agente nunca vio.
Intervenciones (misma evaluacion y semillas que scripts/evaluate_world2024_20261010.py, parcheada en memoria):
    --mode nodnf    entorno: se quitan los -1; el rival sigue corriendo con su ultimo neumatico (sin mas paradas)
    --mode dnfobs   solo observacion: el entorno no cambia, pero el agente ve al rival retirado con su ultimo
                    time_diff_to_agent antes de abandonar (congelado) en vez de +500
    ... --mode <m> <args de evaluate_world2024_20261010.py>
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
i = sys.argv.index('--mode'); MODE = sys.argv[i + 1]; del sys.argv[i:i + 2]
assert MODE in ('nodnf', 'dnfobs'), MODE
import numpy as np
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers

if MODE == 'nodnf':
    for e in W.GP_CONFIGS:
        e['rival_strategies'] = {k: {l: c for l, c in v.items() if c != -1} for k, v in e['rival_strategies'].items()}
else:
    _upd, _reset = F1EnvAllDrivers._update_obs_cache, F1EnvAllDrivers.reset

    def reset(self, *a, **k):
        self._last_td = {}
        return _reset(self, *a, **k)

    def upd(self):
        _upd(self)
        last = getattr(self, '_last_td', None)
        if last is None:
            last = self._last_td = {}
        td = self._obs_cache['time_diff_to_agent']
        for j, d in enumerate(self.rival_drivers):
            if d.is_dnf:
                td[j] = last.get(d.name, 0.0)
            else:
                last[d.name] = float(td[j])

    F1EnvAllDrivers.reset, F1EnvAllDrivers._update_obs_cache = reset, upd
print('intervencion:', MODE, flush=True)
W.main()
