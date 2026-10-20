"""Contrafactual (diagnostico, 2026-10-14): misma evaluacion que scripts/evaluate_world2024_20261010.py pero toda
accion de parada del agente desde la vuelta FROM (por defecto 40) se reemplaza por 'seguir'. Comparando con la
corrida normal (mismas semillas) se mide si la parada tardia gana o pierde posiciones en el simulador.
    ... --from-lap 40 <args de evaluate_world2024_20261010.py>
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
FROM = 40
if '--from-lap' in sys.argv:
    i = sys.argv.index('--from-lap'); FROM = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
import evaluate_world2024_20261010 as W
from environment.f1_env_all_drivers import F1EnvAllDrivers
_step = F1EnvAllDrivers.step


def step(self, action):
    if int(self.lap_number) >= FROM and int(action) != 0:
        action = 0
    return _step(self, action)


F1EnvAllDrivers.step = step
W.main()
