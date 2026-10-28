"""2026-10-27 — Validacion de los rivales del simulador (v7 vs v10) antes de entrenar: 40 carreras por GP con SC/VSC
sorteados (modo de entrenamiento), el agente sin paradas. Por evento SC/VSC: fraccion de rivales en carrera (vuelta > 5)
que para en la 1a vuelta; paradas por rival y compuesto elegido. Datos 2021-23 (rivals_v10_20261027.py): parar durante el
evento 0.386 (SC) y 0.220 (VSC).
    F1_SIMULATION_DIR=trained_models/simulation_v10_20261027 python results/sim_v10_20261027/validate_rivals_20261027.py
(con reproduce/env_paper_v3_20261007.sh, F1_RIVAL_START_ALPHA y los envoltorios de juegos v9r2 en el entorno)
Salida: results/sim_v10_20261027/validate_rivals_<arbol>.csv
"""
import os, sys
sys.path[:0] = ['.', 'src', 'scripts']
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS, load_gp_config

loader = ModelLoader(); rows = []; comp = []
tree = os.path.basename(str(config.SIMULATION_DIR).rstrip('/'))
PAPER = {'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
         'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix'}
for e in [c for c in GP_CONFIGS if c['gp_name'] in PAPER]:
    gp, drv = e['gp_name'], e['agent_driver']
    gpc = load_gp_config(gp)
    env = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True,
                                    initial_positions=gpc['initial_positions_list_2024'], loader=loader, verbose=False,
                                    fast_rivals=True)
    orig = env._simulate_rival_pits
    log = []
    def wrapped(is_yf, _o=orig, _e=env, _l=log):
        a = _o(is_yf)
        alive = [i for i, d in enumerate(_e.rival_drivers) if not d.is_dnf]
        _l.append(dict(lap=_e.lap_number, sc=int(_e.sc > 0), vsc=int(_e.vsc > 0), first=int(bool(_e._is_first_yf_lap)),
                       n_alive=len(alive), n_pit=int(sum(1 for i in alive if a[i] != 0)),
                       comps=[int(a[i]) for i in alive if a[i] != 0]))
        return a
    env._simulate_rival_pits = wrapped
    for s in range(40):
        log.clear()
        env.reset(seed=1000 + s)
        done = False; env.step(2)
        while env.lap_number <= env.n_laps:
            env.step(0)
        L = pd.DataFrame(log)
        ev = L[(L.sc + L.vsc > 0) & (L.first == 1) & (L.lap > 5)]
        for r in ev.itertuples():
            rows.append(dict(tree=tree, gp=gp, race=s, kind='SC' if r.sc else 'VSC', frac=r.n_pit / max(r.n_alive, 1)))
        rows.append(dict(tree=tree, gp=gp, race=s, kind='stops_per_rival', frac=L.n_pit.sum() / 19))
        comp += [dict(tree=tree, gp=gp, c=c) for cs in L.comps for c in cs]
R = pd.DataFrame(rows)
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'validate_rivals_{tree}.csv'), index=False)
print(tree); print(R.groupby('kind').frac.agg(['mean', 'size']).round(3).to_string())
print(R[R.kind != 'stops_per_rival'].groupby(['gp', 'kind']).frac.mean().unstack().round(2).to_string())
