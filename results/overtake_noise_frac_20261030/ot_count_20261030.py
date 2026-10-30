"""2026-10-30 — Adelantamientos por tramo de gap y distribucion de gaps en el simulador multi-piloto, con las opciones
de adelantamiento tomadas del entorno (F1_OVERTAKE_MAX_GAP, F1_OVERTAKE_MEAN_GAP, F1_OVERTAKE_NOISE_FRAC,
F1_FOLLOWING_GAP_SCALE). Misma medicion que results/sim_v10eGQ_20261028/overtake_diag_20261029/ot_sim_phys.py (variante
tree:W): el agente sigue una parada fija a mitad de carrera, se excluyen sus pares, vueltas 5+ sin SC/VSC.
    TAG=l050 N_EP=30 python scripts/gp_configs_fix_20261028.py results/overtake_noise_frac_20261030/ot_count_20261030.py
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
N_EP = int(os.environ.get('N_EP', 30)); TAG = os.environ['TAG']


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
        start = {d.name: (d.position, d.total_time) for d in act}
        order = sorted(act, key=lambda d: d.position)
        rec['yf'] = None
        a = 0 if lap != round(0.5 * N) else hard
        _, _, term, trunc, _ = env.step(a); done = term or trunc; lap += 1
        if rec['yf'] is not False or lap < 5:
            continue
        pos = {d.name: d.position for d in env.drivers}; dnf = {d.name for d in env.drivers if d.is_dnf}
        for k in range(1, len(order)):
            x, y = order[k], order[k - 1]
            if x.name in dnf or y.name in dnf or env.agent_driver.name in (x.name, y.name):
                continue
            rows.append((gp, TAG, seed, lap, start[x.name][1] - start[y.name][1], int(pos[x.name] < pos[y.name])))
    return rows


LOADER = ModelLoader()
IP = {}
for gp in AG:
    with open(os.path.join(str(config.SIMULATION_DIR), gp + ' Grand Prix', 'config.json')) as fh:
        IP[gp] = json.load(fh).get('initial_positions_list_2024')
print('opciones', {k: os.environ.get(k) for k in ('F1_OVERTAKE_MAX_GAP', 'F1_OVERTAKE_MEAN_GAP', 'F1_OVERTAKE_NOISE_FRAC',
                                                   'F1_FOLLOWING_GAP_SCALE', 'F1_SIMULATION_DIR')}, flush=True)
allr = []; t0 = time.time()
for gp in AG:
    for s in range(N_EP):
        allr += run(gp, 1000 + s)
    print(gp, f'{time.time() - t0:.0f}s', flush=True)
R = pd.DataFrame(allr, columns=['GP', 'var', 'seed', 'lap', 'gap', 'ot'])
os.makedirs(os.path.join(HERE, 'ot_count'), exist_ok=True)
R.to_pickle(os.path.join(HERE, 'ot_count', f'{TAG}.pkl')); print('ok', len(R))
