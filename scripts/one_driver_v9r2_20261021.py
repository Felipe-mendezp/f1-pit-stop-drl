"""Mono-piloto v9r con la observacion 'tyre_sets' escalada — 2026-10-21. DESCARTADO (22-oct): no arregla el aprendizaje
(diagnostico B, results/onedriver_v9r2_20261021/README.md); v9r2 = juegos reutilizables, scripts/tyre_reuse_v9r2_20261022.py.

En v9r el mono-piloto recibia 'tyre_sets' sin normalizar (juegos que quedan 0-5, edades 0-21; el entrenamiento usa
VecNormalize(norm_obs=False)) y los agentes on-policy aprendieron a parar en la vuelta 6
(results/onedriver_v9r_20261021/README.md). Fija F1_ONE_DRIVER_SETS_SCALE=1 (juegos / 8, edades / 20; ver
src/environment/f1_env_one_driver.py) y ejecuta el envoltorio v9 con el mono-piloto v8:
    python scripts/one_driver_v9r2_20261021.py {train|screen|fig5|m1} <args de one_driver_v8_20261019.py>
Los agentes v9r (sin escalar) no sirven con esta variable, ni los v9r2 sin ella.
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_ONE_DRIVER_SETS_SCALE', '1')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v9r2 mono-piloto] F1_ONE_DRIVER_SETS_SCALE =', os.environ['F1_ONE_DRIVER_SETS_SCALE'], flush=True)
    sys.argv = [os.path.join(_ROOT, 'scripts', 'tyre_sets_v9_20261021.py'),
                os.path.join(_ROOT, 'scripts', 'one_driver_v8_20261019.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
