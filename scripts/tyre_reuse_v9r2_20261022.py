"""v9r2 = v9r + juegos reutilizables (F1_TYRE_REUSE=1, idea del autor) — 2026-10-22.

El juego que se saca en una parada vuelve al inventario con su edad y se puede volver a montar; al parar se monta el
menos gastado del compuesto (puede ser el mismo que se saca). El agente observa 'tyre_sets' = edad / 20 del juego que se
montaria por compuesto, sin el numero de juegos que quedan (ver src/tyre_sets.py y
results/onedriver_v9r2_20261021/README.md). Fija F1_TYRE_REUSE=1 y ejecuta el envoltorio v9 (inventario por reglamento):
    multi-piloto: python scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py <script.py> <args>
    mono-piloto:  python scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/one_driver_v8_20261019.py {train|screen|fig5|...} <args>
Cambia la observacion: los agentes v9r no sirven en v9r2 ni al reves. La seleccion de semillas, las figuras y las
evaluaciones de los agentes v9r2 deben correr con este envoltorio.
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_TYRE_REUSE', '1')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v9r2] F1_TYRE_REUSE =', os.environ['F1_TYRE_REUSE'], flush=True)
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
