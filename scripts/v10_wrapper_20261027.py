"""Envoltorio del multi-piloto v10 — 2026-10-27 (docs/PROPUESTA_V10_20261027.md, raiz del proyecto).

Fija F1_VSC_FIRST_LAP_ONLY=1 (bajo VSC el agente solo para en la 1a vuelta del periodo, como los rivales; tambien la
QMIP) y, con F1_VEC_DUMMY=1, reemplaza SubprocVecEnv por DummyVecEnv (todos los entornos en el proceso principal: mismo
buffer y mismas semillas, mucho menos CPU y RAM; asi cada tarea cabe en 4 CPU, regla del autor del 27-oct).
El simulador v10 (arbol trained_models/simulation_v10_20261027) llega por F1_SIMULATION_DIR (columna SIM de las tareas).
    python scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py \\
        scripts/rival_start_v8_20261018.py <script.py> <args>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_VSC_FIRST_LAP_ONLY', '1')
if os.environ.get('F1_VEC_DUMMY') == '1':
    import stable_baselines3.common.vec_env as _V
    _V.SubprocVecEnv = _V.DummyVecEnv
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v10] F1_VSC_FIRST_LAP_ONLY =', os.environ['F1_VSC_FIRST_LAP_ONLY'], '| F1_VEC_DUMMY =',
          os.environ.get('F1_VEC_DUMMY', '0'), '| sim:', os.environ.get('F1_SIMULATION_DIR'), flush=True)
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
