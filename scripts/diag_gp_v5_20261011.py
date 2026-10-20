import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-11 — Diagnostico por GP v5 (generaliza scripts/diag_hungarian_v5_20261011.py a cualquier GP del paper
y agrega --world 2024: el entorno simula con neumaticos opcion3 2021-24, agentes y QMIP creen 2021-23,
como scripts/evaluate_world2024_20261010.py).

Corre PPO, RecurrentPPO, QMIP y estrategias fijas en el GP bajo 4 "mundos" que separan
los dos ingredientes que distinguen la Fig. 6 de la Fig. 7:
  rivales  : 'stoch' (logit, como Fig. 6) o 'real' (estrategias 2024, como Fig. 7)
  amarillas: 'stoch' (escenarios SC/VSC pre-generados como en Fig. 6), 'none' o 'real' (vueltas SC/VSC 2024)
Registra por carrera: compuesto inicial, vueltas y compuestos de cada parada, vueltas SC/VSC,
posicion final. No escribe en trained_models/. Salida: results/gp_diag_v5_20261011/<model_dir>/.
    source reproduce/env_paper_v3_20261007.sh
    F1_SIMULATION_DIR=trained_models/simulation_v5_20261010 \
    F1_RL_AGENTS_DIR=trained_models/rl_agents_v5_20261010/selected \
    python scripts/diag_gp_v5_20261011.py --gp Miami_OCO --world 2024 --rivals real --yf real --policies PPO RecurrentPPO QMIP
Estrategia fija: 'fix:<c0>-<lap>:<c1>[-<lap>:<c2>]' con compuestos S/M/H, p. ej. fix:M-30:H.
"""

import argparse, json
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import evaluate_qmip_vs_drl as E
from evaluate_all_gps import GP_CONFIGS, load_gp_config, build_rival_strategies, ALGO_CLASSES
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios
from environment.f1_env_all_drivers import F1EnvAllDrivers, create_f1_env_all_drivers
from model_loader import ModelLoader
import evaluate_world2024_20261010 as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, 'results', 'gp_diag_v5_20261011')
GP = CFG = OUT = None
NAME = {'S': 'Soft', 'M': 'Medium', 'H': 'Hard'}

# --- traza por vuelta del agente ------------------------------------------------
_TRACE, _DONE = [], []
_orig_reset, _orig_step = F1EnvAllDrivers.reset, F1EnvAllDrivers.step
def _reset(self, *a, **kw):
    global _TRACE
    if _TRACE: _DONE.append(_TRACE)
    _TRACE = []
    return _orig_reset(self, *a, **kw)
def _step(self, action):
    lap = self.lap_number
    out = _orig_step(self, action)
    _TRACE.append((lap, int(self.agent_driver.compound), int(self.sc > 0), int(self.vsc > 0), int(self.agent_position),
                   float(self.agent_driver.tyrelife)))
    return out
F1EnvAllDrivers.reset, F1EnvAllDrivers.step = _reset, _step

def last_trace():
    return _TRACE if _TRACE else _DONE[-1]

def summarize(trace, final_pos, changed):
    # parada = cambio de compuesto o neumatico nuevo (tyrelife reinicia): cuenta tambien S->S
    comp, age = [t[1] for t in trace], [t[5] for t in trace]
    stops = [(trace[i][0], comp[i]) for i in range(1, len(comp)) if comp[i] != comp[i - 1] or age[i] < age[i - 1]]
    n_cmp = sum(comp[i] != comp[i - 1] for i in range(1, len(comp)))
    return dict(start=comp[0], n_stops=len(stops), n_stops_compound=n_cmp, stops=' '.join(f'{l}:{c}' for l, c in stops),
                sc_laps=' '.join(str(t[0]) for t in trace if t[2]), vsc_laps=' '.join(str(t[0]) for t in trace if t[3]),
                any_yf=int(any(t[2] or t[3] for t in trace)),
                pos_trace=' '.join(str(t[4]) for t in trace),
                final_position=20 if not changed else final_pos)

# --- un mundo -------------------------------------------------------------------
def configure(raw_env, rivals, laps_vsc, laps_sc, init_pos):
    if rivals == 'real':
        ra, ic = build_rival_strategies(CFG)
        raw_env.set_evaluation_mode(True, laps_vsc, laps_sc, init_pos, ra, ic)
    else:
        raw_env.set_evaluation_mode(True, laps_vsc, laps_sc, init_pos, {})

def seed_all(raw_env, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed); raw_env.action_space.seed(seed)

def run_drl(algo, raw_env, scen, rivals, init_pos, base_seed):
    d = os.path.join(str(config.RL_AGENTS_DIR), CFG['model_dir'], algo)
    vec = VecNormalize.load(os.path.join(d, 'vecnormalize_stats.pkl'), DummyVecEnv([lambda: raw_env]))
    vec.training = False; vec.norm_reward = False
    model = ALGO_CLASSES[algo].load(os.path.join(d, 'best_model.zip'), env=vec)
    rows = []
    for i, (lv, ls) in enumerate(scen):
        configure(raw_env, rivals, lv, ls, init_pos); seed_all(raw_env, base_seed + i)
        obs = vec.reset(); st = None; ep = np.ones((1,), dtype=bool); done = False; info = [{}]
        while not done:
            if algo == 'RecurrentPPO':
                a, st = model.predict(obs, state=st, episode_start=ep, deterministic=True); ep[:] = False
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, r, dn, info = vec.step(a); done = dn[0]
        rows.append(dict(sim=i, **summarize(last_trace(), info[0].get('final_position'), info[0].get('has_changed_compound', False))))
    return rows

def run_raw(policy, raw_env, scen, rivals, init_pos, base_seed, qp):
    avail = qp['available_compounds']; q2c = qp['qmip_to_compound']; c2q = qp['compound_to_qmip']
    act = lambda name: avail.index(q2c[name]) + 1
    if policy.startswith('fix:'):
        parts = policy[4:].split('-')
        plan = {int(p.split(':')[0]): act(NAME[p.split(':')[1]]) for p in parts[1:]}
    rows = []
    for i, (lv, ls) in enumerate(scen):
        configure(raw_env, rivals, lv, ls, init_pos); seed_all(raw_env, base_seed + i)
        raw_env.reset()
        if policy == 'QMIP':
            a0, t0 = E.qmip_choose_start_compound(qp['betas_dict'], qp['time_stop'], raw_env.n_laps, q2c, avail, p=3)
            tprev = [t0]
        else:
            a0 = act(NAME[parts[0]])
        _, _, term, trunc, info = raw_env.step(a0); done = term or trunc
        cur = raw_env.agent_driver.compound
        while not done:
            if policy == 'QMIP':
                g, yf = E.qmip_gamma(raw_env)
                a = E.qmip_choose_action(qp['betas_dict'], qp['time_stop'], raw_env.laps_left,
                                         c2q.get(raw_env.agent_driver.compound, 'Medium'), raw_env.agent_driver.tyrelife,
                                         tprev, raw_env.n_laps, g, q2c, avail, p=3, yf_now=yf,
                                         time_stop_yf=qp.get('time_stop_yf'))
            else:
                a = plan.get(raw_env.lap_number, 0)
            _, _, term, trunc, info = raw_env.step(a); done = term or trunc
            if policy == 'QMIP' and raw_env.agent_driver.compound != cur:
                cur = raw_env.agent_driver.compound; n = c2q.get(cur, 'Medium')
                if n not in tprev: tprev.append(n)
        rows.append(dict(sim=i, **summarize(last_trace(), info.get('final_position'), info.get('has_changed_compound', False))))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rivals', choices=['stoch', 'real'], required=True)
    ap.add_argument('--yf', choices=['stoch', 'none', 'real'], required=True)
    ap.add_argument('--gp', required=True, help='model_dir, p. ej. Miami_OCO')
    ap.add_argument('--world', choices=['2024', 'none'], default='2024')
    ap.add_argument('--policies', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    global GP, CFG, OUT
    CFG = next(c for c in GP_CONFIGS if c['model_dir'] == args.gp); GP = CFG['gp_name']
    OUT = os.path.join(OUT0, args.gp); os.makedirs(OUT, exist_ok=True)
    loader = ModelLoader(); gm = loader.load(GP)
    changed = W.register(GP, gm.laptime_clear, args.world)   # el entorno aplica el mundo en cada reset
    print(f'mundo {args.world}: {len(changed)} coeficientes cambiados', flush=True)
    init_pos = load_gp_config(GP)['initial_positions_list_2024']
    scen = pregenenerate_race_scenarios(args.n, gm.n_laps, gm.vsc_prob, gm.sc_prob, args.seed * 1000)
    if args.yf == 'none': scen = [([], [])] * args.n
    if args.yf == 'real':
        gc = load_gp_config(GP); scen = [(gc['laps_vsc_2024'], gc['laps_sc_2024'])] * args.n
    qp = E.build_qmip_params(loader, GP)   # parcheado por W: la QMIP planifica con 2021-23
    raw_env = create_f1_env_all_drivers(gp=GP, driver=CFG['agent_driver'], deterministic=False, yf_enabled=True,
                                        initial_positions=init_pos, loader=loader, verbose=False)
    for pol in args.policies:
        rows = run_drl(pol, raw_env, scen, args.rivals, init_pos, args.seed) if pol in ('PPO', 'RecurrentPPO') \
            else run_raw(pol, raw_env, scen, args.rivals, init_pos, args.seed, qp)
        df = pd.DataFrame(rows).assign(policy=pol, rivals=args.rivals, yf=args.yf, world=args.world)
        fn = os.path.join(OUT, f'w{args.world}_r{args.rivals}_y{args.yf}{args.tag}__{pol.replace(":", "_")}.csv')
        df.to_csv(fn, index=False)
        print(f'{pol:18s} rivals={args.rivals} yf={args.yf}: pos {df.final_position.mean():.2f}  '
              f'stops {df.n_stops.value_counts(normalize=True).round(2).to_dict()}  '
              f'top plan {df.assign(p=df.start.astype(str) + "|" + df.stops).p.value_counts().head(3).to_dict()}', flush=True)
    print('available', qp['available_compounds'], 'qmip_to_compound', qp['qmip_to_compound'])

if __name__ == '__main__':
    main()
