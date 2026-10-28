"""v10eGQ — adelantamiento y gap de seguimiento corregidos (v10f) — 2026-10-29.

Envoltorio del multi-piloto (como los demas de la cadena; va primero). Fija:
  F1_OVERTAKE_MAX_GAP=inf   el logit decide toda batalla en pista libre (antes solo con gap inicial <= 1.0 s; fuera
                            de eso, adelanta <=> gap predicho < 0);
  F1_OVERTAKE_MEAN_GAP=1    el logit se evalua en el gap con los tiempos de vuelta MEDIOS (sin el ruido de la Ec. 1),
                            igual que se estimo (results/sim_v10eGQ_20261028/overtake_v10f_20261029.py);
  F1_FOLLOWING_GAP_SCALE=0.26 y --following-gap-scale 0.26: el auto que no adelanta queda a 0.4 + Exp(media 0.26) s
                            (antes Exp(1)), calibrado con los gaps reales de autos bloqueados (mediana 0.62 s).
El arbol trained_models/simulation_v10eGQ_20261028 trae el logit nuevo. Diagnostico y pruebas:
results/sim_v10eGQ_20261028/README_adelantamiento_20261029.md. Si el comando trae --following-gap-scale con otro
valor, se reemplaza por 0.26 (y se avisa), para que toda corrida bajo este envoltorio use la misma fisica.
    python scripts/overtake_v10f_20261029.py scripts/tyre_ages_v10eGQ_20261028.py scripts/v10_wrapper_20261027.py ...
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FGS = '0.26'
os.environ['F1_OVERTAKE_MAX_GAP'] = 'inf'
os.environ['F1_OVERTAKE_MEAN_GAP'] = '1'
os.environ['F1_FOLLOWING_GAP_SCALE'] = FGS
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    args = sys.argv[1:]
    for i, a in enumerate(args[:-1]):
        if a == '--following-gap-scale' and args[i + 1] != FGS:
            print(f'[v10f] --following-gap-scale {args[i + 1]} -> {FGS}', flush=True)
            args[i + 1] = FGS
    print('[v10f] F1_OVERTAKE_MAX_GAP = inf | F1_OVERTAKE_MEAN_GAP = 1 | F1_FOLLOWING_GAP_SCALE =', FGS, flush=True)
    target = args[0] if os.path.isabs(args[0]) else os.path.join(_ROOT, args[0])
    sys.argv = [target] + args[1:]
    runpy.run_path(target, run_name='__main__')
