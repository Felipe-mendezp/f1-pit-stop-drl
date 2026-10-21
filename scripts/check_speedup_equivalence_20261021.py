"""2026-10-21 — Equivalence check of the simulator speed-up (src/reg.py plan cache, NumPy softmax compound choice).

Runs 4,000 steps per GP with fixed seeds and actions twice: with the previous code paths
(FastLinearPredictor._build_design_vector_legacy and the pandas compound choice) and with the new ones, and checks
that observations, rewards, cumulative times and rivals' compounds are identical. Prints the speed of each.
    source reproduce/env_paper_v3_20261007.sh
    export F1_SIMULATION_DIR=$PWD/trained_models/simulation_v7_20261015 F1_LAP0_ACTION0_MEDIUM=1 F1_LAP0_PENALTY=-1000
    python scripts/rival_start_v8_20261018.py scripts/check_speedup_equivalence_20261021.py
"""
import os, sys, time, json
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
import numpy as np
import reg
from model_loader import ModelLoader
from environment.f1_env_all_drivers import create_f1_env_all_drivers
import models.cond_logit_compound as CL
NEW = reg.FastLinearPredictor._build_design_vector
OLD = reg.FastLinearPredictor._build_design_vector_legacy
loader = ModelLoader()
def make(gp, drv):
    cfg = json.load(open(os.path.join(os.environ['F1_SIMULATION_DIR'], gp, 'config.json')))
    e = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True,
            initial_positions=cfg.get('initial_positions_list_2024'), loader=loader, verbose=False,
            reward_type='mix', fast_rivals=True)
    e.following_gap_scale = 1.0; e.sc_bunching = True; e.sc_bunch_delta = 1.5
    return e
def traj(gp, drv, n, impl, softmax):
    reg.FastLinearPredictor._build_design_vector = impl
    CL.CondLogitCompoundView.choice_rule = 'softmax' if softmax else None
    e = make(gp, drv); rng = np.random.default_rng(7); out = []
    obs, _ = e.reset(seed=11); t = time.time()
    for k in range(n):
        a = int(rng.choice(4, p=[.94, .02, .02, .02]))
        obs, r, d, tr, _ = e.step(a)
        out.append(np.concatenate([np.ravel(np.asarray(v, dtype=np.float64)) for v in obs.values()] + [[r],
                   [dd.total_time for dd in [e.agent_driver] + e.rival_drivers], [dd.compound for dd in e.rival_drivers]]))
        if d or tr: obs, _ = e.reset()
    return np.array(out), time.time() - t
stops = 0
for gp, drv in [('Bahrain Grand Prix', 'Driver_ALO'), ('Saudi Arabian Grand Prix', 'Driver_NOR'), ('Singapore Grand Prix', 'Driver_HUL'), ('Hungarian Grand Prix', 'Driver_TSU')]:
    a, ta = traj(gp, drv, 4000, OLD, False); b, tb = traj(gp, drv, 4000, NEW, True)
    nst = int((np.diff(a[:, -19:], axis=0) != 0).sum())
    print(f'{gp:26s} identico: {np.array_equal(a, b)}  cambios de compuesto de rivales: {nst:4d}  antes {4000/ta:5.0f} pasos/s  ahora {4000/tb:5.0f} pasos/s  x{ta/tb:.2f}')
