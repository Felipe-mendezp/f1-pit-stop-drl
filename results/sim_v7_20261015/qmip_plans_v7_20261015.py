"""2026-10-15 — Estrategia optima de la QMIP desde la largada (sin SC/VSC) en v6 y v7, por GP.
Mismos parametros que scripts/evaluate_qmip_vs_drl.build_qmip_params (coeficientes de laptime_clear del arbol).
Salida: results/sim_v7_20261015/qmip_plans_v7_20261015.csv
"""
import os, sys
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()
import pandas as pd
from model_loader import ModelLoader
from evaluate_qmip_vs_drl import build_qmip_params
from optimization.qmip import solve_MIP_GUROBI_start, Tires

GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
rows = []
for tree in ['simulation_v6_20261014', 'simulation_v7_20261015']:
    loader = ModelLoader(str(config.MODELS_DIR / tree))
    for gp in GPS:
        q = build_qmip_params(loader, gp)
        n = loader.load(gp).n_laps
        x, y, obj, t0 = solve_MIP_GUROBI_start(q['betas_dict'], q['time_stop'], n=n, w_current=0, t_previous=[], p=3,
                                              gamma=0, N=n, verbose=False)
        seq = [(t0, int(round(x[0])))] + [(Tires[i % 3], int(round(v))) for i, v in enumerate(x[1:]) if round(v) > 0]
        rows.append(dict(arbol=tree[11:13], GP=gp.replace(' Grand Prix', ''), vueltas=n, paradas=len(seq) - 1,
                         plan=' - '.join(f'{c[0]}{l}' for c, l in seq), tiempo=round(obj, 2),
                         **{f'{k}_b0': round(v[0], 3) for k, v in q['betas_dict'].items()},
                         **{f'{k}_b1': round(v[1], 4) for k, v in q['betas_dict'].items()},
                         time_stop=round(q['time_stop'], 2)))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qmip_plans_v7_20261015.csv'), index=False)
pd.set_option('display.width', 250)
print(D.pivot_table(index='GP', columns='arbol', values=['plan', 'paradas'], aggfunc='first').to_string())
print(D[['arbol', 'GP', 'Soft_b0', 'Medium_b0', 'Hard_b0', 'Soft_b1', 'Medium_b1', 'Hard_b1', 'time_stop']].to_string(index=False))
