import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Train RecurrentPPO with different reward types for reward structure comparison.

Trains RecurrentPPO on 3 GPs x 3 reward types = 9 runs.
Models are saved to rl_agents_reward/ to keep existing rl_agents/ intact.

The 'mix' reward type is already trained (in rl_agents/), so this script
trains 'time', 'position', and 'points'.

Usage:
    python scripts/train_reward_comparison.py
"""

import argparse
import torch
from sb3_contrib import RecurrentPPO

from train_rl_agents import (
    create_env_factory,
    create_fresh_envs,
    train_algorithm,
    get_initial_positions,
    seed_all,
    TRAINING_CONFIG,
)
from hyperparams.config import RECURRENT_PPO_CONFIG
from model_loader import ModelLoader

# Configure PyTorch for M3 Pro optimization
torch.set_num_threads(8)

# Output directory (separate from rl_agents/ to avoid overwriting)
MODELS_REWARD_DIR = str(config.RL_REWARD_DIR)

# GP and driver combinations to train
GP_DRIVER_PAIRS = [
    ('Bahrain Grand Prix', 'Driver_ALO'),
    ('Miami Grand Prix', 'Driver_OCO'),
    ('Dutch Grand Prix', 'Driver_RUS'),
]

# Reward types to train (mix is already trained in rl_agents/)
REWARD_TYPES = ['time', 'position', 'points']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RecurrentPPO under different reward types.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility. Default: 42.')
    parser.add_argument('--gps', nargs='+', default=None,
                        help='Subset of GP names to train (default: all pairs). '
                             'E.g. --gps "Miami Grand Prix"')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps per run (default: config).')
    parser.add_argument('--fast-env', action='store_true',
                        help='Numpy fast path for rival/overtake logits '
                             '(validated at init, ~3x faster env steps).')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Override number of parallel envs (default: config).')
    parser.add_argument('--torch-threads', type=int, default=None,
                        help='Override torch.set_num_threads (module default 8).')
    # --- Opt-ins (2026-09-11, regeneración de la Sección P bajo la física del paper);
    # defaults reproducen el protocolo original (física legacy, 4 rewards, árbol RL_REWARD_DIR).
    parser.add_argument('--reward-types', nargs='+', default=None, choices=REWARD_TYPES,
                        help='Subconjunto de rewards a entrenar (default: los 4). Las semillas se '
                             'asignan por el índice en la lista completa, así que un subconjunto '
                             'reproduce las mismas semillas que la corrida completa.')
    parser.add_argument('--output-dir', default=None,
                        help='Árbol de salida (default: RL_REWARD_DIR). Layout <GP>_<DRV>/RecurrentPPO_<reward>/')
    parser.add_argument('--following-gap-scale', type=float, default=0.0)
    parser.add_argument('--sc-bunching', action='store_true')
    parser.add_argument('--sc-bunch-delta', type=float, default=1.5)
    parser.add_argument('--checkpoint-freq', type=int, default=0)
    args = parser.parse_args()

    seed_all(args.seed)

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    if args.gps:
        GP_DRIVER_PAIRS = [(gp, drv) for gp, drv in GP_DRIVER_PAIRS
                           if gp in args.gps]
        if not GP_DRIVER_PAIRS:
            raise SystemExit(f"--gps {args.gps} matched no configured pair")

    n_envs = args.n_envs or TRAINING_CONFIG['n_envs']
    total_timesteps = args.timesteps or TRAINING_CONFIG['total_timesteps']
    eval_freq = TRAINING_CONFIG['eval_freq']
    n_eval_episodes = TRAINING_CONFIG['n_eval_episodes']
    early_stop_patience = TRAINING_CONFIG['early_stop_patience']
    base_seed = args.seed

    total_runs = len(GP_DRIVER_PAIRS) * len(REWARD_TYPES)

    print("\n" + "=" * 70)
    print("REWARD COMPARISON TRAINING - RecurrentPPO")
    print("=" * 70)
    print(f"\n  GPs: {len(GP_DRIVER_PAIRS)}")
    for gp, driver in GP_DRIVER_PAIRS:
        print(f"    - {gp} ({driver})")
    print(f"  Reward types: {', '.join(REWARD_TYPES)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Timesteps per run: {total_timesteps:,}")
    print(f"  Output: {MODELS_REWARD_DIR}")

    loader = ModelLoader()
    run_idx = 0
    print(f"  Base seed: {base_seed}")

    for gp_idx, (gp, agent_driver) in enumerate(GP_DRIVER_PAIRS):
        print("\n" + "=" * 70)
        print(f"LOADING MODELS FOR {gp.upper()}")
        print("=" * 70)

        models = loader.load(gp)
        n_laps = models.n_laps

        print(f"  Laps: {n_laps}")
        print(f"  Compounds: {models.compounds}")

        # Compute dynamic parameters
        dynamic_batch_size = n_laps * n_envs * 2
        dynamic_n_steps = n_laps * 2

        print(f"  batch_size: {dynamic_batch_size}, n_steps: {dynamic_n_steps}")

        # Get initial positions
        initial_positions = get_initial_positions(gp)
        if initial_positions:
            print(f"  Using 2024 initial positions")

        # Output directory for this GP+driver
        gp_short = gp.replace(' Grand Prix', '').replace(' ', '_')
        agent_short = agent_driver.replace('Driver_', '')
        output_dir = os.path.join(args.output_dir or MODELS_REWARD_DIR, f"{gp_short}_{agent_short}")
        os.makedirs(output_dir, exist_ok=True)

        for reward_type in (args.reward_types or REWARD_TYPES):
            rt_idx = REWARD_TYPES.index(reward_type)
            run_idx += 1
            algo_name = f"RecurrentPPO_{reward_type}"
            # Distinct seed per (GP, reward_type) so they don't share RNG state
            run_seed = base_seed + gp_idx * len(REWARD_TYPES) + rt_idx

            print("\n" + "=" * 70)
            print(f"RUN {run_idx}/{total_runs}: {gp_short}_{agent_short} | {algo_name} | seed={run_seed}")
            print("=" * 70)

            # Create env factory with this reward type
            env_factory = create_env_factory(
                gp, agent_driver, loader, initial_positions,
                reward_type=reward_type,
                fast_rivals=args.fast_env,
                following_gap_scale=args.following_gap_scale,
                sc_bunching=args.sc_bunching,
                sc_bunch_delta=args.sc_bunch_delta,
            )

            train_algorithm(
                algo_name=algo_name,
                algo_cls=RecurrentPPO,
                algo_config=RECURRENT_PPO_CONFIG,
                env_factory=env_factory,
                n_envs=n_envs,
                output_dir=output_dir,
                dynamic_batch_size=dynamic_batch_size,
                dynamic_n_steps=dynamic_n_steps,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                early_stop_patience=early_stop_patience,
                seed=run_seed,
                checkpoint_freq=args.checkpoint_freq,
            )

        print("\n" + "=" * 70)
        print(f"COMPLETED {gp.upper()} - all reward types trained")
        print("=" * 70)

    # Final summary
    print("\n" + "=" * 70)
    print("ALL REWARD COMPARISON TRAINING COMPLETED")
    print("=" * 70)
    print(f"\n  Runs completed: {run_idx}/{total_runs}")
    print(f"  Models saved to: {MODELS_REWARD_DIR}")
    for gp, driver in GP_DRIVER_PAIRS:
        gp_short = gp.replace(' Grand Prix', '').replace(' ', '_')
        agent_short = driver.replace('Driver_', '')
        print(f"    {gp_short}_{agent_short}/")
        for rt in REWARD_TYPES:
            print(f"      RecurrentPPO_{rt}/")
    print("=" * 70)
