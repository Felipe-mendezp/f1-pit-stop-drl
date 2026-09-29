import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
Print the pit laps of the DETERMINISTIC prefix for one GP/driver/agent set.

Answers "does this policy stop at all, and where?" without running any
counterfactual rollout. Used to accept or reject a freshly trained model before
spending hours on decision-support sweeps: a policy whose only compound change
is the mandatory stop forced by the rule mask on lap n_laps-1 is degenerate and
has no showcase lap.

Usage:
    python scripts/diag_prefix_pits.py --gp 'Saudi Arabian Grand Prix' \
        --driver Driver_NOR --model-dir Saudi_Arabian_NOR --real-2024
Last line of stdout is machine-readable:  pits=[13, 26] n_laps=44
"""

import argparse
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import (load_gp_config, build_rival_strategies, GP_CONFIGS,
                              ALGO_CLASSES, NORM_OBS_KEYS)
from decision_support_rollouts import run_prefix_multi, action_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gp', required=True)
    ap.add_argument('--driver', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--prefix-seed', type=int, default=0)
    ap.add_argument('--real-2024', action='store_true')
    ap.add_argument('--real-yf', action='store_true')
    # Same opt-in env-physics fixes as decision_support_rollouts.py: a model
    # trained under the corrected physics must be probed under it too.
    ap.add_argument('--pit-det-pass', action='store_true')
    ap.add_argument('--following-gap-scale', type=float, default=0.0)
    args = ap.parse_args()

    entry = next((e for e in GP_CONFIGS
                  if e['gp_name'] == args.gp and e['agent_driver'] == args.driver), None)
    if entry is None:
        raise SystemExit(f"No GP_CONFIGS entry for {args.gp} / {args.driver}")
    gp_config = load_gp_config(args.gp)
    rival_actions, initial_compounds = build_rival_strategies(entry)
    ip = gp_config['initial_positions_list_2024']

    raw_env = create_f1_env_all_drivers(
        gp=args.gp, driver=args.driver, deterministic=False, yf_enabled=True,
        initial_positions=ip, loader=ModelLoader(), verbose=False, fast_rivals=True)
    raw_env.pit_pass_deterministic = args.pit_det_pass
    raw_env.following_gap_scale = args.following_gap_scale
    if args.real_2024:
        raw_env.set_evaluation_mode(
            eval_mode=True, laps_vsc=gp_config['laps_vsc_2024'],
            laps_sc=gp_config['laps_sc_2024'], initial_positions=ip,
            rival_actions=rival_actions, rival_initial_compounds=initial_compounds)
    elif args.real_yf:
        raw_env.set_evaluation_mode(
            eval_mode=True, laps_vsc=gp_config['laps_vsc_2024'],
            laps_sc=gp_config['laps_sc_2024'], initial_positions=ip, rival_actions={})

    vec_env = DummyVecEnv([lambda: raw_env])
    model_path = os.path.join(str(config.RL_AGENTS_DIR), args.model_dir,
                              'RecurrentPPO', 'best_model.zip')
    vecnorm = os.path.join(os.path.dirname(model_path), 'vecnormalize_stats.pkl')
    if os.path.exists(vecnorm):
        vec_env = VecNormalize.load(vecnorm, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS,
                               clip_reward=10000.0)
    vec_env.training = False
    vec_env.norm_reward = False
    model = ALGO_CLASSES['RecurrentPPO'].load(model_path, env=vec_env)

    n_laps = raw_env.n_laps
    caps = run_prefix_multi(model, vec_env, raw_env, [n_laps - 1],
                            initial_compounds, args.prefix_seed)
    hist = caps[n_laps - 1]['history']
    labels = action_labels(raw_env.available_compounds)
    # lap 0 "pit" is the starting-compound choice, not a stop.
    pits = [(l, c) for l, c, p in hist if p and l >= 1]
    start = next((c for l, c, _ in hist if l == 0), None)

    print(f"model: {model_path}")
    print(f"labels: {labels}")
    print(f"start compound: C{start}")
    for l, c in pits:
        print(f"  stop lap {l} -> C{c}")
    print(f"pits={[l for l, _ in pits]} n_laps={n_laps}")


if __name__ == '__main__':
    main()
