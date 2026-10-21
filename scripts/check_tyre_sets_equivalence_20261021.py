"""2026-10-21 — Equivalence check of the tyre-set change (opt-in F1_TYRE_SETS) with the flag OFF.

Runs fixed-seed, fixed-action trajectories of the v8 environments and stores (save) or compares (compare) them:
  - multi-driver, simulated rivals (training mode), 4 GPs x 3,000 steps;
  - multi-driver, evaluation mode with the real 2024 rival strategies (Fig. 7 protocol), 2 GPs x 3 races;
  - single-driver environment (Fig. 5), 3 GPs x 3,000 steps.
With F1_TYRE_SETS unset the 'compare' run must be bit-identical to the 'save' run made before the change. With the
flag on, 'smoke' runs the same trajectories and prints tyre-set statistics (no comparison).
    source reproduce/env_paper.sh
    export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015
    python scripts/rival_start_v8_20261018.py scripts/check_tyre_sets_equivalence_20261021.py {save|compare|smoke} <npz>
"""
import os, sys, json, pickle
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
import numpy as np
from model_loader import ModelLoader
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
import evaluate_all_gps as A

mode, path = sys.argv[1], sys.argv[2]
loader = ModelLoader()
SIM = os.environ['F1_SIMULATION_DIR']


def make(gp, drv):
    cfg = json.load(open(os.path.join(SIM, gp, 'config.json')))
    e = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True,
                                  initial_positions=cfg.get('initial_positions_list_2024'), loader=loader,
                                  verbose=False, reward_type='mix', fast_rivals=True)
    e.following_gap_scale = 1.0; e.sc_bunching = True; e.sc_bunch_delta = 1.5
    return e


def row(e, obs, r):
    return np.concatenate([np.ravel(np.asarray(v, dtype=np.float64)) for k, v in sorted(obs.items())
                           if k != 'tyre_sets'] + [[r],
                          [d.total_time for d in [e.agent_driver] + e.rival_drivers],
                          [d.compound for d in e.rival_drivers], [d.tyrelife for d in e.rival_drivers]])


def multi(gp, drv, n):
    e = make(gp, drv); rng = np.random.default_rng(7); out = []
    obs, _ = e.reset(seed=11)
    for _ in range(n):
        a = int(rng.choice(4, p=[.94, .02, .02, .02]))
        obs, r, d, tr, _ = e.step(a); out.append(row(e, obs, r))
        if d or tr:
            obs, _ = e.reset()
    return np.array(out), e


def multi_eval(entry, races):
    e = make(entry['gp_name'], entry['agent_driver'])
    strat, init = A.build_rival_strategies(entry)
    e.set_evaluation_mode(rival_actions=strat, rival_initial_compounds=init)
    rng = np.random.default_rng(3); out = []
    for k in range(races):
        obs, _ = e.reset(seed=100 + k); d = False
        while not d:
            a = int(rng.choice(4, p=[.94, .02, .02, .02]))
            obs, r, d, tr, _ = e.step(a); out.append(row(e, obs, r))
    return np.array(out), e


def one(gp, n):
    import train_one_driver_paperhp_20260924 as T
    comps, n_laps = T.PAPER_GPS[gp]
    pred = pickle.load(open(os.path.join(SIM, f'{gp}_fast_predictor.pkl'), 'rb')) \
        if os.path.exists(os.path.join(SIM, f'{gp}_fast_predictor.pkl')) else loader.load(gp).laptime_clear
    e = F1EnvOneDriverV2Optimized(n_laps=n_laps, available_compounds=comps, gp=gp, driver='VER', model_laptime=pred)
    rng = np.random.default_rng(5); out = []
    obs, _ = e.reset(seed=13)
    for _ in range(n):
        a = int(rng.choice(4, p=[.94, .02, .02, .02]))
        obs, r, d, tr, _ = e.step(a)
        out.append(np.array([obs[k] for k in sorted(obs) if k != 'tyre_sets'] + [r, e.total_time], dtype=np.float64))
        if d or tr:
            obs, _ = e.reset()
    return np.array(out), e


res = {}
for gp, drv in [('Bahrain Grand Prix', 'Driver_ALO'), ('Saudi Arabian Grand Prix', 'Driver_NOR'),
                ('Singapore Grand Prix', 'Driver_HUL'), ('Hungarian Grand Prix', 'Driver_TSU')]:
    res[f'multi|{gp}'], env = multi(gp, drv, 3000)
for entry in [c for c in A.GP_CONFIGS if c['model_dir'] in ('Bahrain_ALO', 'Singapore_HUL')]:
    res[f'eval|{entry["gp_name"]}'], env = multi_eval(entry, 3)
for gp in ['Bahrain Grand Prix', 'Miami Grand Prix', 'Dutch Grand Prix']:
    res[f'one|{gp}'], env = one(gp, 3000)

if mode == 'save':
    np.savez(path, **res); print('guardado', path, {k: v.shape for k, v in res.items()})
elif mode == 'compare':
    ref = np.load(path)
    ok = True
    for k, v in res.items():
        same = ref[k].shape == v.shape and np.array_equal(ref[k], v)
        ok &= same
        print(f'{k:45s} identico: {same}')
    print('TODO IDENTICO' if ok else 'HAY DIFERENCIAS')
else:
    for k, v in res.items():
        print(f'{k:45s} pasos {len(v)}  recompensa media {v[:, -1 if k.startswith("one") else 0].mean():.3f}')
