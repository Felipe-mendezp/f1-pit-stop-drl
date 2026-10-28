"""v10e (edad efectiva; F1_V10E=G|GQ; GQ adoptado el 28-oct, usar scripts/one_driver_v10eGQ_20261028.py) — 2026-10-28, copia de one_driver_v10d_20261028.py. Mono-piloto v10d (Fig. 5 / Tabla N.1): como scripts/one_driver_v10_20261027.py con el arbol v10d (compuestos corregidos con OpenF1 y el live timing; copia de one_driver_v10c_20261028.py).

Igual que scripts/one_driver_v8_20261019.py (penalizaciones del multi-piloto, accion 0 en la vuelta 0 = medio, sin filtro,
hiperparametros publicados), pero con el arbol trained_models/simulation_onedriver_v10_20261027 (docs/PROPUESTA_V10_20261027.md).
one_driver_v7_20261015.py fija su arbol en el codigo; aqui se ejecuta con ese arbol reemplazado por el de v10 (sin
modificar el archivo). Mismos subcomandos y argumentos:
    python scripts/one_driver_v10d_20261028.py {verify|train|screen|fig5|m1} <args de one_driver_v7_20261015.py>
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
from one_driver_v8_20261019 import V8          # fija las variables de recompensa v8 (setdefault) al importarse
OLD = "TREE = os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_v7_20261015')"
NEW = "TREE = os.path.join(_ROOT, 'trained_models', 'simulation_onedriver_v10e' + os.environ['F1_V10E'] + '_20261028')"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v10e mono-piloto]', '  '.join(f'{k}={os.environ[k]}' for k in V8), flush=True)
    path = os.path.join(_ROOT, 'scripts', 'one_driver_v7_20261015.py')
    src = open(path).read()
    assert src.count(OLD) == 1, 'cambio one_driver_v7_20261015.py'
    sys.argv[0] = path
    exec(compile(src.replace(OLD, NEW), path, 'exec'), {'__name__': '__main__', '__file__': path})
