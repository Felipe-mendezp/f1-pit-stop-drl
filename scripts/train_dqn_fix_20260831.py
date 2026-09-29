import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Variante de DQN con exploration_final_eps de CLI (2026-08-31).

Análogo exacto del fix de PPO (train_ppo_fix_20260820.py): el v1 deja un 5%
de exploración residual PARA SIEMPRE (exploration_final_eps=0.05, el mismo
"ruido de dosis" que el ent_coef=0.05 de PPO). El estudio v2 ya midió que
0.02 mejora DDQN 8.89→7.43 bajo la física legacy. Este wrapper entrena DQN
(DoubleDQN) con el v1 EXACTO salvo el override de eps, sin tocar
hyperparams/config.py ni train_rl_agents.py.

Uso:
  python scripts/train_dqn_fix_20260831.py --eps-final 0.02 \
      --seeds 42 --fast-env --torch-threads 4 --following-gap-scale 1.0 \
      --sc-bunching --out-root trained_models/rl_agents_yf2024_dqnfix_20260831
"""

import argparse
import copy

import torch

from train_rl_agents import (
    AVAILABLE_ALGORITHMS,
    create_env_factory,
    train_algorithm,
    get_initial_positions,
    seed_all,
    TRAINING_CONFIG,
)
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS

GP_DRIVER = {e['gp_name']: e['agent_driver'] for e in GP_CONFIGS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eps-final', type=float, required=True)
    ap.add_argument('--gps', nargs='+', default=list(GP_DRIVER))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--timesteps', type=int, default=None)
    ap.add_argument('--n-envs', type=int, default=None)
    ap.add_argument('--torch-threads', type=int, default=None)
    ap.add_argument('--fast-env', action='store_true')
    ap.add_argument('--following-gap-scale', type=float, default=0.0)
    ap.add_argument('--sc-bunching', action='store_true')
    ap.add_argument('--sc-bunch-delta', type=float, default=1.5)
    ap.add_argument('--out-root', required=True)
    args = ap.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    algo_cls, base_cfg = AVAILABLE_ALGORITHMS['DQN']
    cfg = copy.deepcopy(base_cfg)
    cfg['exploration_final_eps'] = args.eps_final

    n_envs = args.n_envs or TRAINING_CONFIG['n_envs']
    total_timesteps = args.timesteps or TRAINING_CONFIG['total_timesteps']
    loader = ModelLoader()

    print(f"DQN-FIX: exploration_final_eps={cfg['exploration_final_eps']} "
          f"-> {args.out_root}")
    for gp in args.gps:
        driver = GP_DRIVER[gp]
        models = loader.load(gp)
        n_laps = models.n_laps
        gp_short = gp.replace(' Grand Prix', '').replace(' ', '_')
        agent_short = driver.replace('Driver_', '')
        for seed in args.seeds:
            out_dir = os.path.join(args.out_root, f'{gp_short}_{agent_short}')
            os.makedirs(out_dir, exist_ok=True)
            print(f'\n== DQN | {gp} | seed {seed} -> {out_dir}')
            seed_all(seed)
            env_factory = create_env_factory(
                gp, driver, loader, get_initial_positions(gp), reward_type='mix',
                fast_rivals=args.fast_env,
                following_gap_scale=args.following_gap_scale,
                sc_bunching=args.sc_bunching,
                sc_bunch_delta=args.sc_bunch_delta)
            train_algorithm(
                algo_name='DQN',
                algo_cls=algo_cls,
                algo_config=cfg,
                env_factory=env_factory,
                n_envs=n_envs,
                output_dir=out_dir,
                dynamic_batch_size=n_laps * n_envs * 2,
                dynamic_n_steps=n_laps * 2,
                total_timesteps=total_timesteps,
                eval_freq=TRAINING_CONFIG['eval_freq'],
                n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
                early_stop_patience=TRAINING_CONFIG['early_stop_patience'],
                seed=seed,
            )


if __name__ == '__main__':
    main()
