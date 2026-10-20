"""Multi-piloto v8 = v7 + compuestos de partida de los rivales Dirichlet-multinomial simetrica — 2026-10-18.

Fija F1_RIVAL_START_ALPHA = alpha de data/rival_start_alpha_2021_2023_20261018.json (si no viene dado; ajuste en
scripts/fit_rival_start_alpha_20261018.py) y ejecuta el script pedido a traves de scripts/rival_dnf_v6_20261014.py
(abandonos de rivales + pilotos v5hd, igual que v7). Afecta solo a los rivales sin compuesto real (entrenamiento,
seleccion de semillas, Fig. 6); en la Fig. 7 los compuestos de partida siguen siendo los reales de 2024.
2026-10-18 (tarde): ademas F1_LAP0_ACTION0_MEDIUM=1 y F1_LAP0_PENALTY=-1000 (si no vienen dados): la accion 0 en la
vuelta 0 da el compuesto medio en vez de uno sorteado, con penalizacion -1000 en vez de -600 (decision del autor).
    python scripts/rival_start_v8_20261018.py <script.py> <args del script>
"""
import json, os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get('F1_RIVAL_START_ALPHA'):
    with open(os.path.join(_ROOT, 'data', 'rival_start_alpha_2021_2023_20261018.json')) as _f:
        os.environ['F1_RIVAL_START_ALPHA'] = str(json.load(_f)['alpha'])
assert float(os.environ['F1_RIVAL_START_ALPHA']) > 0
os.environ.setdefault('F1_LAP0_ACTION0_MEDIUM', '1')
os.environ.setdefault('F1_LAP0_PENALTY', '-1000')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v8] F1_RIVAL_START_ALPHA =', os.environ['F1_RIVAL_START_ALPHA'], ' F1_LAP0_ACTION0_MEDIUM =',
          os.environ['F1_LAP0_ACTION0_MEDIUM'], ' F1_LAP0_PENALTY =', os.environ['F1_LAP0_PENALTY'], flush=True)
    sys.argv = [os.path.join(_ROOT, 'scripts', 'rival_dnf_v6_20261014.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
