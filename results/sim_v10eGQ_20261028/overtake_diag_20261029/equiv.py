"""Con F1_OVERTAKE_* sin fijar, el entorno nuevo = respaldo previo (bit a bit); ademas cuenta llamadas con flags."""
import os, sys, json, importlib.util
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
import numpy as np
import environment.f1_env_all_drivers as NEW
spec = importlib.util.spec_from_file_location('f1env_old', os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/f1env_bak_ot_20261029.py'))
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
from model_loader import ModelLoader
L = ModelLoader()
AG = {'Bahrain': 'ALO', 'Singapore': 'HUL', 'Belgian': 'GAS'}
def traj(M, gp, seed):
    with open(os.path.join(str(config.SIMULATION_DIR), gp + ' Grand Prix', 'config.json')) as fh: ip = json.load(fh).get('initial_positions_list_2024')
    env = M.create_f1_env_all_drivers(gp + ' Grand Prix', 'Driver_' + AG[gp], deterministic=False, yf_enabled=True, initial_positions=ip,
                                      loader=L, verbose=False, reward_type='mix', fast_rivals=True)
    env.following_gap_scale = float(os.environ['F1_FOLLOWING_GAP_SCALE']); env.sc_bunching = True; env.sc_bunch_delta = 1.5
    env.reset(seed=seed); out = []; done = False; lap = 0
    hard = 1 + list(env.available_compounds).index('HARD') if 'HARD' in env.available_compounds else 1
    while not done:
        _, r, te, tr, _ = env.step(hard if lap == round(0.5 * env.n_laps) else 0); done = te or tr; lap += 1
        out.append([r] + [d.total_time for d in env.drivers] + [d.position for d in env.drivers])
    return np.array(out)
same = True
for gp in AG:
    for seed in range(5):
        a, b = traj(NEW, gp, seed), traj(OLD, gp, seed)
        same &= a.shape == b.shape and np.array_equal(a, b)
print('flags off: nuevo == anterior en 15 carreras:', same)
