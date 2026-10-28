"""Prueba en el simulador multi-piloto: adelantamientos por carrera con la regla actual vs (a) sin umbral + logit C,
(b) logit en el gap medio (sin ruido de tiempo de vuelta). Se corre con la misma cadena de envoltorios del entrenamiento."""
import os, sys, json, time
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import environment.f1_env_all_drivers as F
from model_loader import ModelLoader
T = os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
P = pd.read_csv(T + 'ot_support.csv'); P = P[P.v == 'C'].set_index('GP')
AG = {'Bahrain': 'ALO', 'Belgian': 'GAS', 'Dutch': 'STR', 'Hungarian': 'TSU', 'Miami': 'OCO', 'Saudi Arabian': 'NOR',
      'Singapore': 'HUL', 'United States': 'MAG'}
N_EP = int(os.environ.get('N_EP', 30)); VARS = os.environ.get('VARS', 'actual:E0,ab:E0').split(',')
import inspect, textwrap
_src = inspect.getsource(F.F1EnvAllDrivers._resolve_overtakes)
_old = '''                    pred_laptimes[driver_k.name] = max(
                        pred_laptimes[driver_k.name],
                        pred_laptimes[driver_k_1.name] - gap_k_to_k1
                        + self._sample_following_gap()
                    )
                    break'''
assert _old in _src
_new = '''                    if pred_gap < MIN_GAP:     # E2: piso solo si el de atras quedaria a menos de MIN_GAP (bloqueado)
                        pred_laptimes[driver_k.name] = max(
                            pred_laptimes[driver_k.name],
                            pred_laptimes[driver_k_1.name] - gap_k_to_k1
                            + self._sample_following_gap()
                        )
                    break'''
_ns = F.__dict__; exec(textwrap.dedent(_src.replace(_old, _new)), _ns); RO_E2 = _ns['_resolve_overtakes']
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class Logit:                                   # logit de 2 parametros con la interfaz que usa el entorno
    def __init__(s, a, b): s.a, s.b = a, b
    def predict(s, X): return sig(s.a + s.b * np.asarray(X['delta_total'], float))


class MeanProxy:                               # guarda la media predicha (antes del ruido) de cada piloto
    def __init__(s, inner): s._i, s.mean = inner, {}
    def __getattr__(s, k): return getattr(s._i, k)
    def predict_single_with_std(s, d):
        m, sd = s._i.predict_single_with_std(d); s.mean[id(d)] = m; return m, sd
    def predict_single(s, d):
        m = s._i.predict_single(d); s.mean[id(d)] = m; return m


def setup(env, gp, var):
    if var in ('a', 'ab'):
        a, b = P.loc[gp, 'const'], P.loc[gp, 'slope']
        env.logit_overtake = Logit(a, b); env._ot_intercept, env._ot_slope = a, b
    if var in ('b', 'ab'):
        px = MeanProxy(env.fast_predictor_clear); env.fast_predictor_clear = px
        def batch(sorted_drivers, ott, pred, env=env, px=px):
            # todos los pares ordenados (incluye los que se forman tras un adelantamiento en la misma vuelta)
            mu = {d.name: px.mean.get(id(d.lap_state), pred[d.name]) for d in sorted_drivers}
            HIT[0] += sum(id(d.lap_state) in px.mean for d in sorted_drivers); HIT[1] += len(sorted_drivers)
            out = {}
            for x in sorted_drivers:
                for y in sorted_drivers:
                    if x is not y:
                        out[(x.name, y.name)] = sig(env._ot_intercept + env._ot_slope * (ott[x.name] - ott[y.name] + mu[x.name] - mu[y.name]))
            return out
        env._compute_overtake_probs_batch = batch


def run(gp, spec, seed):
    var, phys = spec.split(':')
    F.OVERTAKE_MAX_GAP = np.inf if var in ('a', 'ab') else 1.0
    env = F.create_f1_env_all_drivers(gp + ' Grand Prix', 'Driver_' + AG[gp], deterministic=(phys == 'E3'), yf_enabled=True,
                                      initial_positions=IP[gp], loader=LOADER, verbose=False, reward_type='mix', fast_rivals=True)
    env.following_gap_scale = float(os.environ['F1_FOLLOWING_GAP_SCALE']) if phys == 'W' else {'E1': 0.0, 'E4': 0.26, 'E5': 0.26}.get(phys, 1.0); env.sc_bunching = True; env.sc_bunch_delta = 1.5
    setup(env, gp, var)
    if phys == 'E6':
        env._sample_following_gap = lambda: 0.66
    if phys == 'E7':
        _bg = np.load(T + 'otsim/blocked_gaps.npy')
        env._sample_following_gap = lambda env=env: float(_bg[env.np_random.integers(len(_bg))])
    if phys in ('E2', 'E5'):
        env._resolve_overtakes = RO_E2.__get__(env)
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
            g = start[x.name][1] - start[y.name][1]
            rows.append((gp, spec, seed, lap, g, int(pos[x.name] < pos[y.name])))
    return rows


HIT = [0, 0]
LOADER = ModelLoader()
IP = {}
for gp in AG:
    with open(os.path.join(str(config.SIMULATION_DIR), gp + ' Grand Prix', 'config.json')) as fh:
        IP[gp] = json.load(fh).get('initial_positions_list_2024')
allr = []; t0 = time.time()
for gp in AG:
    for var in VARS:
        for s in range(N_EP):
            allr += run(gp, var, 1000 + s)
    print(gp, f'{time.time() - t0:.0f}s', flush=True)
R = pd.DataFrame(allr, columns=['GP', 'var', 'seed', 'lap', 'gap', 'ot'])
R.to_pickle(T + os.environ.get('OUT', 'otsim/ot_sim_phys.pkl')); print('ok', len(R), '| medias encontradas', HIT)
