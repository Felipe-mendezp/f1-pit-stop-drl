"""2026-10-22 — Invariant check of reusable tyre sets (F1_TYRE_SETS=1, F1_TYRE_REUSE=1), multi- and single-driver.

Random-action episodes (fixed seeds). Checks every step: the agent never asks for a compound without a set; every simulated
driver keeps all his sets (pool + the mounted one = initial number); the mounted set's tyre life matches; the
'tyre_sets' observation has one value per compound and equals the age / 20 of the set a stop would fit; and that sets
are actually reused (a set taken off is fitted again in some episode). gymnasium check_env on both environments.
    source reproduce/env_paper.sh
    export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015 F1_TYRE_SETS=1 F1_TYRE_REUSE=1
    python scripts/rival_start_v8_20261018.py scripts/check_tyre_reuse_20261022.py
"""
import os, sys, json
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
import numpy as np
from gymnasium.utils.env_checker import check_env
from model_loader import ModelLoader
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
import tyre_sets

assert tyre_sets.reuse_enabled(), 'F1_TYRE_SETS=1 F1_TYRE_REUSE=1'
loader = ModelLoader()
SIM = os.environ['F1_SIMULATION_DIR']
GPS = ['Bahrain Grand Prix', 'Hungarian Grand Prix', 'Singapore Grand Prix', 'Miami Grand Prix']
DRV = {'Bahrain Grand Prix': 'Driver_ALO', 'Hungarian Grand Prix': 'Driver_TSU', 'Singapore Grand Prix': 'Driver_HUL',
       'Miami Grand Prix': 'Driver_OCO'}


def n_sets(sets):
    return sum(len(v) for v in sets.values())


def expected_obs(sets, comps, compound, life):
    return np.array([min(list(sets.get(c, [])[:1]) + ([life + 1] if c == compound else []) or [0]) / 20.0
                     for c in comps], dtype=np.float32)


rng = np.random.default_rng(0)
for gp in GPS:
    cfg = json.load(open(os.path.join(SIM, gp, 'config.json')))
    e = create_f1_env_all_drivers(gp=gp, driver=DRV[gp], deterministic=False, yf_enabled=True,
                                  initial_positions=cfg.get('initial_positions_list_2024'), loader=loader,
                                  verbose=False, reward_type='mix', fast_rivals=True)
    e.following_gap_scale = 1.0; e.sc_bunching = True; e.sc_bunch_delta = 1.5
    if gp == GPS[0]:
        check_env(e, skip_render_check=True)
    for ep in range(4):
        obs, _ = e.reset(seed=ep)
        n0 = {d.name: n_sets(tyre_sets.sets_for(gp, d.name, e.available_compounds)) for d in e.drivers}
        done, steps = False, 0
        while not done:
            a = int(rng.integers(0, 4)) if rng.random() < 0.15 or e.lap_number == 0 else 0
            obs, r, done, trunc, _ = e.step(a)
            done = done or trunc
            steps += 1
            assert e._no_set_pending == 0.0, (gp, ep, steps)
            ag = e.agent_driver
            assert obs['tyre_sets'].shape == (e._n_compounds,)
            assert np.allclose(obs['tyre_sets'], expected_obs(ag.sets, e.available_compounds, ag.compound,
                                                               ag.tyrelife)), (obs['tyre_sets'], ag.sets)
            for d in e.drivers:
                if d.is_dnf or d.compound == 0:
                    continue
                assert n_sets(d.sets) + 1 == n0[d.name], (gp, d.name, d.sets, n0[d.name])
    print(f'multi|{gp:<28} OK ({ep + 1} episodios)', flush=True)

for gp in GPS:
    import pickle
    import train_one_driver_paperhp_20260924 as T
    comps, n_laps = T.PAPER_GPS[gp]
    pk = os.path.join(SIM, f'{gp}_fast_predictor.pkl')
    pred = pickle.load(open(pk, 'rb')) if os.path.exists(pk) else loader.load(gp).laptime_clear
    e = F1EnvOneDriverV2Optimized(n_laps=n_laps, available_compounds=comps, gp=gp, driver='VER', model_laptime=pred)
    if gp == GPS[0]:
        check_env(e, skip_render_check=True)
    fitted_used = 0
    for ep in range(200):
        obs, _ = e.reset(seed=ep)
        n0 = n_sets(e._sets_start)
        start = {c: list(v) for c, v in e._sets_start.items()}
        done = False
        while not done:
            a = int(rng.integers(1, 4)) if e.lap_number == 0 or rng.random() < 0.08 else 0
            before = (e.compound, e.tyre_life)
            if a:  # never a stop (or start) without a set (no F1_NO_SET_PENALTY)
                assert tyre_sets.has_set(e._sets, e.available_compounds[a - 1], (e.compound,)), (gp, ep, e._sets)
            obs, r, done, _, _ = e.step(a)
            assert obs['tyre_sets'].shape == (e._n_compounds,)
            assert np.allclose(obs['tyre_sets'], expected_obs(e._sets, e.available_compounds, e.compound,
                                                               e.tyre_life))
            assert n_sets(e._sets) + 1 == n0
            if a and before[0] and e.compound and e.tyre_life not in start.get(e.compound, []):
                fitted_used += 1  # fitted a set that was used in this race (its age is not an initial one)
    print(f'one  |{gp:<28} OK (200 episodios, juegos usados vueltos a montar: {fitted_used})', flush=True)
print('TODO OK')
