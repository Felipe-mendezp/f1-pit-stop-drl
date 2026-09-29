import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Single-driver (QMIP benchmark, Fig. 5 / Table N.1) trainer with the PAPER
hyper-parameters — 2026-09-24.

Why a second trainer: `train_one_driver.py` was ported from the experimental
`Code/F1_one_driver/train_optimized.py` (Nov-2025: reward normalisation OFF,
Tanh, n_envs 8, small buffers). The checkpoints behind the paper's Fig. 5 and
Table N.1 (`Code/F1_one_driver/models_V0`, Dec-2025) were produced by the
notebook `test.ipynb` (cell 12) with a DIFFERENT configuration, the one listed
in the (commented) Table F.1 "without competition":
  n_envs = 10, 5M steps, ReLU, VecNormalize(norm_reward=True), eval every 1000
  vec-steps (RecurrentPPO: 10000) with 100 episodes, and per algorithm:
  DQN  [64,64,64], buffer 3M, batch 2*n_l*n_e (lr 1e-4 default)
  A2C  [64,64], lr 3e-4, n_steps n_l, ent 0.05, gae 0.95, RMSprop
  TRPO [64,64,64], n_steps n_l, batch n_l*n_e (lr 1e-3 default)
  PPO  [64,64,64], lr 3e-4, n_steps 2*n_l, batch n_l*n_e, 10 epochs, clip 0.2,
       ent 0.01, target_kl 0.015
  RPPO LSTM 1x32 + [32,32] (no critic LSTM), lr 5e-4, n_steps n_l, batch
       n_l*n_e, ent 0.10, clip 0.30, 10 epochs
Evaluated deterministically, those checkpoints reproduce Table N.1 to the
third decimal (see docs/single_driver_dsq1000_2026-09-24.md).

Kept from the repo trainer: the corrected environment (2026-07-20 fixes),
seeded fresh SubprocVecEnvs per algorithm, DoubleDQN (the paper reports
"Double DQN"), --algos / --seed / --timesteps, output under
F1_RL_AGENTS_DIR/one_driver/<GP>/<algo>/best_model.zip. The two-compound-rule
penalty is taken from F1_ONE_DRIVER_DSQ_PENALTY (env default -100).
"""

import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from sb3_contrib import RecurrentPPO, TRPO

from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
from double_dqn import DoubleDQN

PAPER_GPS = {
    'Bahrain Grand Prix': (['C1', 'C2', 'C3'], 57),
    'Saudi Arabian Grand Prix': (['C2', 'C3', 'C4'], 50),
    'Miami Grand Prix': (['C2', 'C3', 'C4'], 57),
    'Emilia Romagna Grand Prix': (['C3', 'C4', 'C5'], 63),
    'Hungarian Grand Prix': (['C3', 'C4', 'C5'], 70),
    'Belgian Grand Prix': (['C2', 'C3', 'C4'], 44),
    'Dutch Grand Prix': (['C1', 'C2', 'C3'], 72),
    'Singapore Grand Prix': (['C3', 'C4', 'C5'], 62),
    'United States Grand Prix': (['C2', 'C3', 'C4'], 56),
}
ALGOS_ALL = ['RecurrentPPO', 'PPO', 'DQN', 'A2C', 'TRPO']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gps', nargs='+', default=None, help='GP names (default: the 9 paper GPs)')
    ap.add_argument('--algos', nargs='+', default=None, choices=ALGOS_ALL)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timesteps', type=int, default=5_000_000)
    ap.add_argument('--n-envs', type=int, default=10)
    ap.add_argument('--torch-threads', type=int, default=None)
    ap.add_argument('--eval-freq', type=int, default=1_000, help='vec-steps between evaluations (RecurrentPPO uses 10x)')
    ap.add_argument('--n-eval-episodes', type=int, default=100)
    ap.add_argument('--vec', choices=['dummy', 'subproc'], default='dummy',
                    help='Vectorisation. The notebook used make_vec_env default = DummyVecEnv; '
                         'for this microsecond-step env it is also faster and ~20 processes lighter.')
    args = ap.parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    seed_all(args.seed)
    algos = args.algos or ALGOS_ALL
    gps = args.gps or list(PAPER_GPS)
    out_root = str(config.RL_AGENTS_DIR / 'one_driver')
    print(f"output: {out_root}\nDSQ penalty env var: {os.environ.get('F1_ONE_DRIVER_DSQ_PENALTY', '(default -100)')}")

    for gp in gps:
        compounds, n_laps = PAPER_GPS[gp]
        with open(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'), 'rb') as f:
            predictor = pickle.load(f)
        env_kwargs = dict(n_laps=n_laps, available_compounds=compounds, gp=gp,
                          driver='VER', model_laptime=predictor, verbose=False)
        n_envs = args.n_envs
        gp_short = gp[:-11]

        vec_cls = DummyVecEnv if args.vec == 'dummy' else SubprocVecEnv

        def fresh_envs(algo_seed):
            tr = make_vec_env(F1EnvOneDriverV2Optimized, env_kwargs=env_kwargs, n_envs=n_envs,
                              vec_env_cls=vec_cls, seed=algo_seed)
            ev = make_vec_env(F1EnvOneDriverV2Optimized, env_kwargs=env_kwargs, n_envs=n_envs,
                              vec_env_cls=vec_cls, seed=algo_seed + 10_000)
            # Paper configuration: reward normalisation ON (obs are discrete).
            tr = VecNormalize(tr, norm_obs=False, norm_reward=True)
            ev = VecNormalize(ev, norm_obs=False, norm_reward=True)
            sync_envs_normalization(tr, ev)
            return tr, ev

        def evalcb(ev, algo, freq):
            return EvalCallback(ev, best_model_save_path=f"{out_root}/{gp_short}/{algo}",
                                log_path=f"{out_root}/logs/{gp_short}_{algo}/", eval_freq=freq,
                                n_eval_episodes=args.n_eval_episodes, verbose=0)

        specs = {
            'RecurrentPPO': (0, RecurrentPPO, "MultiInputLstmPolicy", dict(
                learning_rate=5e-4, n_steps=n_laps, batch_size=n_laps * n_envs, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.3, ent_coef=0.1,
                policy_kwargs=dict(activation_fn=nn.ReLU, lstm_hidden_size=32, net_arch=[32, 32],
                                   n_lstm_layers=1, enable_critic_lstm=False)), args.eval_freq * 10),
            'PPO': (1, PPO, "MultiInputPolicy", dict(
                learning_rate=3e-4, n_steps=n_laps * 2, batch_size=n_laps * n_envs, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, target_kl=0.015,
                policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[64, 64, 64])), args.eval_freq),
            'DQN': (2, DoubleDQN, "MultiInputPolicy", dict(
                buffer_size=3_000_000, batch_size=n_laps * n_envs * 2,
                policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[64, 64, 64])), args.eval_freq),
            'A2C': (3, A2C, "MultiInputPolicy", dict(
                learning_rate=3e-4, n_steps=n_laps, gamma=0.99, gae_lambda=0.95, ent_coef=0.05,
                vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True, use_rms_prop=True,
                rms_prop_eps=1e-5,
                policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[64, 64])), args.eval_freq),
            'TRPO': (4, TRPO, "MultiInputPolicy", dict(
                n_steps=n_laps, batch_size=n_laps * n_envs,
                policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[64, 64, 64])), args.eval_freq),
        }
        for algo in [a for a in ALGOS_ALL if a in algos]:
            k, cls, policy, kw, freq = specs[algo]
            print(f"\n=== {gp}: training {algo} ({args.timesteps:,} steps, seed {args.seed + k}) ===", flush=True)
            tr, ev = fresh_envs(args.seed + k)
            model = cls(policy, tr, tensorboard_log=f"{out_root}/tensorboard_det/", device="cpu",
                        verbose=1, seed=args.seed + k, **kw)
            model.learn(args.timesteps, callback=evalcb(ev, algo, freq), progress_bar=True,
                        tb_log_name=f"{algo}_{gp_short}")
            tr.close(); ev.close()
        print(f"\nCompleted {gp}", flush=True)
    print("\nALL TRAINING COMPLETED", flush=True)


if __name__ == '__main__':
    main()
