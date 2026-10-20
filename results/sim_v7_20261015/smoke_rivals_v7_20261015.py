"""2026-10-15 — Prueba del entorno con el arbol v7 frente a v6: carga, rutas rapidas y paradas simuladas de los rivales.
El agente nunca para por iniciativa propia (accion 0); se cuentan las paradas de los 19 rivales por carrera (baja de la
vida del neumatico) y el compuesto elegido (rol duro/medio/blando por GP). Sin abandonos de rivales (F1_RIVAL_DNF).
    [FAST=1] python results/sim_v7_20261015/smoke_rivals_v7_20261015.py [N carreras por GP]
"""
import os, sys, time
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
from model_loader import ModelLoader
from environment.f1_env_all_drivers import create_f1_env_all_drivers

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FAST = os.environ.get('FAST', '1') == '1'          # como el entrenamiento (--fast-env)
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
rows = []
for tree in ['simulation_v6_20261014', 'simulation_v7_20261015']:
    loader = ModelLoader(str(config.MODELS_DIR / tree))
    for gp in GPS:
        env = create_f1_env_all_drivers(gp, 'Driver_VER', loader=loader, fast_rivals=FAST)
        comps = sorted(env.gp_compounds)
        role = {c: r for c, r in zip(comps, ['duro', 'medio', 'blando'])}
        t0 = time.time()
        for ep in range(N):
            env.reset(seed=1000 + ep)
            prev = {d.name: (d.tyrelife, d.compound) for d in env.rival_drivers}
            npit = {d.name: 0 for d in env.rival_drivers}; first = {}
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(0)
                done = term or trunc
                for d in env.rival_drivers:
                    tl, c = prev[d.name]
                    if d.tyrelife < tl:
                        npit[d.name] += 1
                        first.setdefault(d.name, env.lap_number / env.n_laps)
                        rows.append(dict(arbol=tree[11:13], GP=gp, kind='comp', rol=role.get(f'C{d.compound}', '?')))
                    prev[d.name] = (d.tyrelife, d.compound)
            for k, v in npit.items():
                rows.append(dict(arbol=tree[11:13], GP=gp, kind='race', paradas=v, primera=first.get(k, np.nan)))
        print(tree, gp, f'rapida parada={env._stop_betas is not None} compuesto={env._choice_betas is not None}',
              f'{(time.time() - t0) / N:.2f} s/carrera', flush=True)
D = pd.DataFrame(rows)
r = D[D.kind == 'race']
pd.set_option('display.width', 200)
print('\nParadas por rival y carrera (media) y fraccion de carrera de la primera parada (media):')
print(r.pivot_table(index='GP', columns='arbol', values=['paradas', 'primera'], aggfunc='mean').round(3).to_string())
print('\nTotal:', r.groupby('arbol').paradas.mean().round(3).to_dict(),
      '| distribucion:', r.groupby('arbol').paradas.value_counts(normalize=True).round(3).unstack().to_dict('index'))
c = D[D.kind == 'comp']
print('\nCompuesto elegido en la parada (proporcion por rol):')
print(c.groupby('arbol').rol.value_counts(normalize=True).round(3).unstack().to_string())
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'smoke_rivals_v7_20261015_fast{int(FAST)}.csv'), index=False)
