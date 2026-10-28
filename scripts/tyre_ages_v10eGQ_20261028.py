"""v10eGQ — edades de los juegos 2024 en vueltas de empuje (edad efectiva GQ) — 2026-10-28.

Fija (si no vienen dadas) F1_TYRE_SETS = data/tyre_sets_2024_reg_v10eGQ_20261028.csv (mismo inventario por reglamento
que data/tyre_sets_2024_reg_20261021.csv, con la edad de cada juego usado en vueltas de empuje) y F1_TYRE_STINT_AGES =
data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv (edad de cada tanda real de 2024 con la misma definicion); ambas de
results/tyrelife_effective_20261028/sim_ages_gq_20261028.py. Debe ir antes de scripts/tyre_sets_v9_20261021.py (que
fija F1_TYRE_SETS con setdefault). Ejecuta el script pedido con el resto de los argumentos:
    python scripts/tyre_ages_v10eGQ_20261028.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py ...
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_TYRE_SETS', os.path.join(_ROOT, 'data', 'tyre_sets_2024_reg_v10eGQ_20261028.csv'))
os.environ.setdefault('F1_TYRE_STINT_AGES', os.path.join(_ROOT, 'data', 'tyre_race_stint_ages_2024_v10eGQ_20261028.csv'))
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v10eGQ] F1_TYRE_SETS =', os.environ['F1_TYRE_SETS'], ' F1_TYRE_STINT_AGES =', os.environ['F1_TYRE_STINT_AGES'],
          flush=True)
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
