import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""2026-11-03 — Juegos de neumaticos disponibles en los puntos de decision de la Fig. 11 (idea del autor: tabla de
apendice). Reconstruye el prefijo real del piloto hasta cada vuelta pedida con la misma cadena que la figura
(gp_configs_fix + ds_rollouts_realhist --run, F1_DS_FORCE=hist, --real-2024, prefix_seed 0) y vuelca, en la vuelta de
decision: el juego montado con su edad efectiva y el pool de juegos restantes por compuesto con sus edades. Con
F1_TYRE_REUSE=1 el juego montado vuelve al pool al parar, asi que una parada puede volver a calzarlo: por eso se
informa aparte.
    python scripts/gp_configs_fix_20261028.py scripts/ds_rollouts_realhist_20261024.py --run \
        scripts/tyre_sets_at_decisions_20261103.py --gp "United States Grand Prix" --driver Driver_MAG \
        --model-dir United_States_MAG --laps 3,17,36 --algo PPO --out <json>
"""
import argparse, json

import numpy as np
import torch

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
if os.environ.get('F1_DS_WORLD2024') == '1':
    import ds_rollouts_world2024_20261011  # noqa: F401  (parchea ModelLoader.load)
from evaluate_all_gps import (load_gp_config, build_rival_strategies, GP_CONFIGS,
                              ALGO_CLASSES, NORM_OBS_KEYS)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from decision_support_rollouts import run_prefix_multi, restore_from, action_labels

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--gp', required=True)
ap.add_argument('--driver', required=True)
ap.add_argument('--model-dir', required=True)
ap.add_argument('--laps', required=True, help='vueltas de decision separadas por coma')
ap.add_argument('--algo', default='PPO')
ap.add_argument('--prefix-seed', type=int, default=0)
ap.add_argument('--following-gap-scale', type=float,
                default=float(os.environ.get('F1_FOLLOWING_GAP_SCALE', '0.0')))
ap.add_argument('--out', required=True)
args = ap.parse_args()
laps = [int(x) for x in args.laps.split(',')]

entry = next(e for e in GP_CONFIGS if e['gp_name'] == args.gp and e['agent_driver'] == args.driver)
gp_config = load_gp_config(args.gp)
rival_actions, initial_compounds = build_rival_strategies(entry)
loader = ModelLoader()
raw_env = create_f1_env_all_drivers(gp=args.gp, driver=args.driver, deterministic=False, yf_enabled=True,
                                    initial_positions=gp_config['initial_positions_list_2024'],
                                    loader=loader, verbose=False)
raw_env.following_gap_scale = args.following_gap_scale
raw_env.set_evaluation_mode(eval_mode=True, laps_vsc=gp_config['laps_vsc_2024'], laps_sc=gp_config['laps_sc_2024'],
                            initial_positions=gp_config['initial_positions_list_2024'],
                            rival_actions=rival_actions)
vec_env = DummyVecEnv([lambda: raw_env])
mp = os.path.join(str(config.RL_AGENTS_DIR), args.model_dir, args.algo, 'best_model.zip')
vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
vec_env = VecNormalize.load(vn, vec_env) if os.path.exists(vn) else \
    VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
vec_env.training = False; vec_env.norm_reward = False
model = ALGO_CLASSES[args.algo].load(mp, env=vec_env)
labels = action_labels(raw_env.available_compounds)

caps = run_prefix_multi(model, vec_env, raw_env, laps, initial_compounds, args.prefix_seed)
out = {'gp': args.gp, 'driver': args.driver, 'algo': args.algo,
       'available_compounds': list(map(int, raw_env.available_compounds)),
       'n_laps': int(raw_env.n_laps), 'laps': {}}
for n in laps:
    env = restore_from(caps[n]['snap'])
    ag = env.agent_driver
    pool = {int(c): sorted(int(a) for a in ag.sets.get(c, [])) for c in env.available_compounds}
    out['laps'][str(n)] = {'mounted': {'compound': int(ag.compound), 'age': int(ag.tyrelife)},
                           'pool': pool, 'position': int(ag.position),
                           'rec': labels[caps[n]['rec_action']]}
    print(f"lap {n}: on C{ag.compound} age {ag.tyrelife}, pool " +
          ' '.join(f'C{c}:{pool[c]}' for c in sorted(pool)) + f", rec {labels[caps[n]['rec_action']]}", flush=True)
json.dump(out, open(args.out, 'w'), indent=1)
print('wrote', args.out)
