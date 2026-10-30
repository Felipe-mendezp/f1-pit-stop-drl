"""2026-10-30 — Pares adyacentes por vuelta en el simulador multi-piloto, con lo necesario para compararlos con la muestra
real del logit (results/sim_v10eGQ_20261028/overtake_diag_20261029): gap al inicio de vuelta, diferencia de tiempos de
vuelta medios (delta_total = gap + mean_x - mean_y), parada de cada auto en la vuelta y si adelanta.
Misma conduccion que ../overtake_noise_frac_20261030/ot_count_20261030.py (agente con una parada fija a mitad de carrera,
se excluyen sus pares). Opciones del entorno desde las variables F1_*.
    TAG=r28k GPS=Hungarian N_EP=100 python scripts/gp_configs_fix_20261028.py results/overtake_calib_20261030/sim_pairs_20261030.py
"""
import os, sys, json, time
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import environment.f1_env_all_drivers as F
from model_loader import ModelLoader
HERE = os.path.dirname(os.path.abspath(__file__))
AG = {'Bahrain': 'ALO', 'Belgian': 'GAS', 'Dutch': 'STR', 'Hungarian': 'TSU', 'Miami': 'OCO', 'Saudi Arabian': 'NOR',
      'Singapore': 'HUL', 'United States': 'MAG'}
N_EP = int(os.environ.get('N_EP', 100)); TAG = os.environ['TAG']
GPS = os.environ['GPS'].split(',') if os.environ.get('GPS') else list(AG)


def run(gp, seed):
    env = F.create_f1_env_all_drivers(gp + ' Grand Prix', 'Driver_' + AG[gp], deterministic=False, yf_enabled=True,
                                      initial_positions=IP[gp], loader=LOADER, verbose=False, reward_type='mix',
                                      fast_rivals=True)
    env.following_gap_scale = float(os.environ['F1_FOLLOWING_GAP_SCALE']); env.sc_bunching = True; env.sc_bunch_delta = 1.5
    rec = {'yf': None}
    orig = env._resolve_overtakes
    def ro(is_yf=False):
        rec['yf'] = is_yf; return orig(is_yf)
    env._resolve_overtakes = ro
    env.reset(seed=seed)
    hard = 1 + list(env.available_compounds).index('HARD') if 'HARD' in env.available_compounds else 1
    N = env.n_laps; rows = []; lap = 0; done = False
    while not done:
        act = [d for d in env._active_drivers]
        start = {d.name: (d.position, d.total_time, d.tyrelife, d.compound, d.stint_laps) for d in act}
        order = sorted(act, key=lambda d: d.position)
        rec['yf'] = None
        a = 0 if lap != round(0.5 * N) else hard
        _, _, term, trunc, _ = env.step(a); done = term or trunc; lap += 1
        if rec['yf'] is not False:
            continue
        D = {d.name: d for d in env.drivers}
        # parada en esta vuelta: stint_laps vuelve a 0 (la edad no sirve con juegos reusados, que pueden ser mas viejos)
        pit = {n: int(D[n].stint_laps == 0 or D[n].compound != v[3]) for n, v in start.items()}
        prev = {n: int(v[4] == 0) for n, v in start.items()}   # paro en la vuelta anterior
        for k in range(1, len(order)):
            x, y = order[k], order[k - 1]
            if D[x.name].is_dnf or D[y.name].is_dnf or env.agent_driver.name in (x.name, y.name):
                continue
            rows.append((gp, seed, lap, N, start[x.name][0], start[x.name][1] - start[y.name][1],
                         D[x.name].mean_laptime - D[y.name].mean_laptime, pit[x.name], pit[y.name], prev[x.name], prev[y.name],
                         int(D[x.name].position < D[y.name].position)))
    return rows


LOADER = ModelLoader()
IP = {}
for gp in GPS:
    with open(os.path.join(str(config.SIMULATION_DIR), gp + ' Grand Prix', 'config.json')) as fh:
        IP[gp] = json.load(fh).get('initial_positions_list_2024')
allr = []; t0 = time.time()
for gp in GPS:
    for s in range(N_EP):
        allr += run(gp, 1000 + s)
    print(gp, f'{time.time() - t0:.0f}s', flush=True)
R = pd.DataFrame(allr, columns=['GP', 'seed', 'lap', 'n_laps', 'pos', 'gap', 'dmean', 'pit_x', 'pit_y', 'prev_x', 'prev_y', 'ot'])
os.makedirs(os.path.join(HERE, 'sim'), exist_ok=True)
R.to_pickle(os.path.join(HERE, 'sim', f'{TAG}__{"_".join(g.replace(" ", "") for g in GPS)}.pkl')); print('ok', len(R))
