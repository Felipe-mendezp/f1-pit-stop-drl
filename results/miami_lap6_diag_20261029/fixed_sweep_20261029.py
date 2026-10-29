"""Diagnostico C (29-oct): estrategias fijas de una parada en Miami OCO, mundo de seleccion (= entrenamiento)."""
import sys, os, time, json, argparse
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT,'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS, load_gp_config, ALGO_CLASSES, NORM_OBS_KEYS
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios
ap = argparse.ArgumentParser()
ap.add_argument('--gp', default='Miami_OCO'); ap.add_argument('--n', type=int, default=100)
ap.add_argument('--laps', default='6,7,8,10,12,15,18,21,24,27,30,33,36,40,45')
ap.add_argument('--agents', default='')   # e.g. PPO:1,DQN:8
ap.add_argument('--time-only', action='store_true'); ap.add_argument('--out', required=True)
ap.add_argument('--part', default='0/1')
ap.add_argument('--tree', default='trained_models/retrained_v10eGQf_20261029/seed{seed:02d}')   # 29-oct B3: p.ej. trained_models/probe_miami_entropy_20261029/ent0.05_seed{seed:02d}
A = ap.parse_args()
e = next(c for c in GP_CONFIGS if c['model_dir'] == A.gp)
gp, drv = e['gp_name'], e['agent_driver']
ip = load_gp_config(gp)['initial_positions_list_2024']
loader = ModelLoader(); gm = loader.load(gp)
scen = pregenenerate_race_scenarios(n_sims=A.n, n_laps=gm.n_laps, race_vsc_prob=gm.vsc_prob,
                                    race_sc_prob=gm.sc_prob, base_seed=2026 * 1000)
env = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True,
                                initial_positions=ip, loader=loader, verbose=False)
comps = list(env.available_compounds); nl = env.n_laps
print('compounds', comps, 'n_laps', nl, flush=True)

def setup(i):
    lv, ls = scen[i]
    env.set_evaluation_mode(eval_mode=True, laps_vsc=lv, laps_sc=ls, initial_positions=ip, rival_actions={})
    s = 2026 + i
    np.random.seed(s); torch.manual_seed(s); env.np_random = np.random.default_rng(s); env.action_space.seed(s)

def fixed(i, plan):
    # plan = {0: arranque, L: compuesto}; si la parada en L no tiene efecto (2a+ vuelta de SC, regla sc_bunching)
    # se reintenta en la vuelta siguiente
    setup(i); env.reset(); R, done, info = 0.0, False, {}
    (L, c), = [(k, v) for k, v in plan.items() if k > 0]; pend = True; actual = None
    while not done:
        lap = int(env.lap_number)
        a = comps.index(plan[0]) + 1 if lap == 0 else (comps.index(c) + 1 if pend and lap >= L else 0)
        _, r, t, tr, info = env.step(a); R += r; done = t or tr
        if a and lap >= 1 and env.agent_driver.compound == c:
            pend = False; actual = lap
    ch = info.get('has_changed_compound', False)
    return R, (info.get('final_position', np.nan) if ch else 20), ch, actual

rows = []
if A.time_only:
    t0 = time.time(); print(fixed(0, {0: comps[1], 6: comps[0]}), fixed(0, {0: comps[1], 5: comps[0]}))
    print('s/race', (time.time() - t0) / 3); sys.exit()
plans = []
for L in map(int, A.laps.split(',')):
    for s in comps:
        for c in comps:
            if c != s: plans.append((f'{s}-{L}-{c}', {0: s, L: c}))
k, K = map(int, A.part.split('/'))
plans = plans[k::K]
t0 = time.time()
for name, plan in plans:
    for i in range(A.n):
        R, pos, ch, act = fixed(i, plan)
        rows.append(dict(strat=name, sim=i, reward=R, pos=pos, ok=ch, pit_lap=act))
    print(name, f"{np.mean([r['reward'] for r in rows if r['strat']==name]):.1f}",
          f"{np.mean([r['pos'] for r in rows if r['strat']==name]):.2f}", f'{time.time()-t0:.0f}s', flush=True)
# agentes
for spec in filter(None, A.agents.split(',')):
    al, sd = spec.split(':')
    mp = os.path.join(A.tree.format(seed=int(sd)), A.gp, al, 'best_model.zip')
    ve = DummyVecEnv([lambda: env]); vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
    ve = VecNormalize.load(vn, ve); ve.training = False; ve.norm_reward = False
    m = ALGO_CLASSES[al].load(mp, env=ve)
    for i in range(A.n):
        setup(i); obs = ve.reset(); done = False; R = 0.0; lstm = None; st = np.ones((1,), bool); pits = []
        while not done:
            lap = int(env.lap_number)
            if al == 'RecurrentPPO':
                a, lstm = m.predict(obs, state=lstm, episode_start=st, deterministic=True); st = np.zeros((1,), bool)
            else:
                a, _ = m.predict(obs, deterministic=True)
            if int(a[0]) > 0: pits.append(f'{lap}:{comps[int(a[0])-1]}')
            obs, r, d, info = ve.step(a); R += float(r[0]); done = d[0]
        ch = info[0].get('has_changed_compound', False)
        rows.append(dict(strat=f'{al}s{sd}', sim=i, reward=R, pos=info[0].get('final_position', np.nan) if ch else 20,
                         ok=ch, pits=' '.join(pits)))
    print(spec, np.mean([r['reward'] for r in rows if r['strat']==f'{al}s{sd}']),
          np.mean([r['pos'] for r in rows if r['strat']==f'{al}s{sd}']), flush=True)
pd.DataFrame(rows).to_csv(A.out, index=False)
