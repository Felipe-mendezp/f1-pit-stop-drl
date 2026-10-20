"""Multi-piloto v6 = v5h + 4 pilotos de mitad de grilla (v5hd) + abandonos de rivales en el simulador — 2026-10-14.

Fija F1_RIVAL_DNF = data/rival_dnf_2021_2023_20261014.json (si no viene dado) y ejecuta el script pedido a traves de
scripts/gp_drivers_v5hd_20261014.py (pilotos v5hd; en Bahrain, Saudi, Miami y Singapore no cambia nada).
Afecta solo a rivales estocasticos (entrenamiento, seleccion de semillas, Fig. 6); con estrategias reales (Fig. 7)
los abandonos son los de 2024. Motivo y diagnostico: results/world2024_v5h_ood_20261014/README.md.
    python scripts/rival_dnf_v6_20261014.py <script.py> <args del script>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('F1_RIVAL_DNF', os.path.join(_ROOT, 'data', 'rival_dnf_2021_2023_20261014.json'))
assert os.path.isfile(os.environ['F1_RIVAL_DNF']), os.environ['F1_RIVAL_DNF']
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v6] F1_RIVAL_DNF =', os.environ['F1_RIVAL_DNF'], flush=True)
    sys.argv = [os.path.join(_ROOT, 'scripts', 'gp_drivers_v5hd_20261014.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
