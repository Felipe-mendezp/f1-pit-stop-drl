"""v9 = v8 + juegos de neumaticos reales de 2024 (F1_TYRE_SETS) — 2026-10-21.

Fija (si no vienen dadas) F1_TYRE_SETS = data/tyre_sets_2024_reg_20261021.csv (juegos al inicio de la carrera con las
devoluciones del reglamento 2024, adoptado por el autor el 21-oct en la noche; edades reales de las tandas de carrera en
data/tyre_race_stint_ages_2024_20261021.csv, ver src/tyre_sets.py) y F1_NO_SET_PENALTY=-1000 (decision del autor), y
ejecuta el script pedido, que normalmente es el envoltorio v8 correspondiente:
    multi-piloto: python scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py <script.py> <args>
    mono-piloto:  python scripts/tyre_sets_v9_20261021.py scripts/one_driver_v8_20261019.py {train|screen|fig5|...} <args>
Todo lo demas (simuladores simulation_v7_20261015 / simulation_onedriver_v7_20261015, hiperparametros, semillas) es igual
que v8. Con las variables cambia el espacio de observacion (clave 'tyre_sets'): los agentes v8 no sirven en v9.
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_TYRE_SETS', os.path.join(_ROOT, 'data', 'tyre_sets_2024_reg_20261021.csv'))
os.environ.setdefault('F1_NO_SET_PENALTY', '-1000')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v9] F1_TYRE_SETS =', os.environ['F1_TYRE_SETS'], ' F1_NO_SET_PENALTY =', os.environ['F1_NO_SET_PENALTY'],
          flush=True)
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
