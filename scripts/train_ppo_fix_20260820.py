import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Fix de PPO bajo la física pit-logit (2026-08-20).

Diagnóstico (logs_boxplots_2026-08-18): PPO v1 queda atrapado en un EQUILIBRIO
DE INDIFERENCIA en 8/9 GPs — entropía alta sostenida (H 0.4-0.6 al final),
clip_fraction ≈ 0, KL ≈ 1e-4, EV = 1.00: la política se queda semi-uniforme
(argmax multi-pit → P20), el crítico ajusta perfectamente esa política
aleatoria, las ventajas son ruido y el gradiente de política se anula mientras
ent_coef=0.05 (2.5× el de RPPO) la empuja de vuelta a la uniformidad.

Este wrapper entrena PPO con el v1 EXACTO salvo overrides de CLI (ent-coef,
lr), sin tocar hyperparams/config.py ni train_rl_agents.py (patrón de
train_rppo_setenc.py).

Uso:
  python scripts/train_ppo_fix_20260820.py --ent-coef 0.02 \
      --gps 'Singapore Grand Prix' 'Bahrain Grand Prix' --seeds 42 \
      --fast-env --torch-threads 4 --following-gap-scale 1.0 \
      --out-root trained_models/rl_agents_ppofix_20260820/entA_s42
"""

import argparse
import copy

import torch

from stable_baselines3 import PPO

from train_rl_agents import (
    create_env_factory,
    train_algorithm,
    get_initial_positions,
    seed_all,
    TRAINING_CONFIG,
)
from hyperparams.config import PPO_CONFIG
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS

GP_DRIVER = {e['gp_name']: e['agent_driver'] for e in GP_CONFIGS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ent-coef', type=float, required=True)
    ap.add_argument('--lr', type=float, default=None,
                    help='None = mantener el 3e-4 del v1')
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

    cfg = copy.deepcopy(PPO_CONFIG)
    cfg['ent_coef'] = args.ent_coef
    if args.lr is not None:
        cfg['learning_rate'] = args.lr

    n_envs = args.n_envs or TRAINING_CONFIG['n_envs']
    total_timesteps = args.timesteps or TRAINING_CONFIG['total_timesteps']
    loader = ModelLoader()

    print(f"PPO-FIX: ent_coef={cfg['ent_coef']} lr={cfg['learning_rate']} "
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
            print(f'\n== PPO | {gp} | seed {seed} -> {out_dir}')
            seed_all(seed)
            env_factory = create_env_factory(
                gp, driver, loader, get_initial_positions(gp), reward_type='mix',
                fast_rivals=args.fast_env,
                following_gap_scale=args.following_gap_scale,
                sc_bunching=args.sc_bunching,
                sc_bunch_delta=args.sc_bunch_delta)
            train_algorithm(
                algo_name='PPO',
                algo_cls=PPO,
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
