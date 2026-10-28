"""2026-10-27 — v10 (arbol simulation_onedriver_v10_20261027 via scripts/one_driver_v10_20261027.py): copia de results/onedriver_v9r2_20261022/screen_v9r2_20261022.py. Original v9r2: copia de results/onedriver_v9r_20261021/screen_v9r_20261021.py con F1_TYRE_REUSE=1
(juegos reutilizables, = scripts/tyre_reuse_v9r2_20261022.py). Con reutilizacion nunca falta juego, asi que NS deberia
quedar vacio (control).

Original: 2026-10-21 — Screening mono-piloto v9r registrando la vuelta 0 y los pedidos de compuesto sin juegos (solo lectura).

Copia de results/onedriver_v8_20261019/screen_lap0_20261019.py para v9r: ejecuta
scripts/tyre_sets_v9_20261021.py scripts/one_driver_v10_20261027.py screen <args> (F1_TYRE_SETS = inventario por
reglamento, F1_NO_SET_PENALTY = -1000, penalizaciones v8) con F1EnvOneDriverV2Optimized envuelto para anotar en LOG_FILE:
    L0 <gp> <accion> C<compuesto> <sin_juego 0/1>     una por episodio (vuelta 0; sin_juego = pidio un compuesto sin juegos)
    NS <gp> <vuelta> <accion>                          cada parada pedida a un compuesto sin juegos (no para, -1000)
En screen_one_driver_20260923.py cada GP corre primero la QMIP y luego DQN, A2C, TRPO, PPO, RecurrentPPO; la gap del
screening es solo tiempo (no incluye las penalizaciones), por eso summarize_v9r_20261021.py cuenta estos eventos.
F1_QMIP_SYMMETRY=full como en la Fig. 5 v8 final (solo afecta a la QMIP sin juegos; con juegos el orden ya es fijo).
    LOG_FILE=<archivo> python results/onedriver_v9r_20261021/screen_v9r_20261021.py --agents-dir <dir> <args de screen>
"""
import os, sys, runpy
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['F1_SIMULATION_DIR'] = os.path.join(ROOT, 'trained_models', 'simulation_onedriver_v10_20261027')
os.environ.setdefault('F1_TYRE_SETS', os.path.join(ROOT, 'data', 'tyre_sets_2024_reg_20261021.csv'))
os.environ.setdefault('F1_NO_SET_PENALTY', '-1000')
os.environ['F1_QMIP_SYMMETRY'] = 'full'
os.environ['F1_TYRE_REUSE'] = '1'
a = sys.argv.index('--agents-dir')
os.environ['F1_RL_AGENTS_DIR'] = os.path.join(ROOT, sys.argv[a + 1])
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized as E  # noqa: E402
from tyre_sets import has_set, reuse_enabled  # noqa: E402
assert reuse_enabled()
LOG = os.environ['LOG_FILE']
open(LOG, 'w').close()
_lap0, _step = E._handle_lap_0, E.step


def _h(self, action):
    n = self._n_compounds
    comp = self.available_compounds[action - 1] if 1 <= action <= n else self.available_compounds[n // 2]
    no_set = int(bool(self._tyre_sets and not self._sets.get(comp)))
    out = _lap0(self, action)
    with open(LOG, 'a') as f:
        f.write(f'L0\t{self.gp}\t{int(action)}\tC{self.compound}\t{no_set}\n')
    return out


def _s(self, action):
    action = int(action)
    # v9r2: con reutilizacion el compuesto montado siempre se puede pedir (su juego vuelve), = env (has_set)
    if self.lap_number > 0 and 1 <= action <= self._n_compounds and self._tyre_sets \
            and not has_set(self._sets, self.available_compounds[action - 1], (self.compound,)):
        with open(LOG, 'a') as f:
            f.write(f'NS\t{self.gp}\t{self.lap_number}\t{action}\n')
    return _step(self, action)


E._handle_lap_0, E.step = _h, _s
sys.argv = [os.path.join(ROOT, 'scripts', 'tyre_sets_v9_20261021.py'),
            os.path.join(ROOT, 'scripts', 'one_driver_v10_20261027.py'), 'screen'] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
