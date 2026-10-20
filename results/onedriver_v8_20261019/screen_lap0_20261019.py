"""2026-10-19 — Screening mono-piloto v8 registrando la accion de la vuelta 0 (solo lectura de modelos).

Ejecuta scripts/one_driver_v8_20261019.py screen <args> con F1EnvOneDriverV2Optimized._handle_lap_0 envuelto para
anotar (gp, accion, compuesto) en el archivo LAP0_LOG, una linea por episodio. En screen_one_driver_20260923.py cada
GP corre primero la QMIP y luego DQN, A2C, TRPO, PPO, RecurrentPPO (los que existen); summarize_lap0_20261019.py
empareja las lineas con las filas del TSV y verifica que el compuesto coincida con el 'start C?' del plan.
F1_SIMULATION_DIR y F1_RL_AGENTS_DIR se fijan antes de importar el entorno (que importa config), como en el wrapper.
    LAP0_LOG=<archivo> python results/onedriver_v8_20261019/screen_lap0_20261019.py --agents-dir <dir> <args de screen>
"""
import os, sys, runpy
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['F1_SIMULATION_DIR'] = os.path.join(ROOT, 'trained_models', 'simulation_onedriver_v7_20261015')
a = sys.argv.index('--agents-dir')
os.environ['F1_RL_AGENTS_DIR'] = os.path.join(ROOT, sys.argv[a + 1])
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized as E  # noqa: E402
LOG = os.environ['LAP0_LOG']
open(LOG, 'w').close()
_orig = E._handle_lap_0


def _h(self, action):
    out = _orig(self, action)
    with open(LOG, 'a') as f:
        f.write(f'{self.gp}\t{int(action)}\tC{self.compound}\n')
    return out


E._handle_lap_0 = _h
sys.argv = [os.path.join(ROOT, 'scripts', 'one_driver_v8_20261019.py'), 'screen'] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
