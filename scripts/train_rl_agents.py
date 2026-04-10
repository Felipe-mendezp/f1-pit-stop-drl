import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
RL Training script for F1 Multi-Driver Environment (All Drivers).

This script trains RL agents in the full 20-driver F1 simulation environment.

Key features:
- Uses F1EnvAllDrivers with 20 drivers (1 agent + 19 rivals)
- Rivals use logit models for pit stop decisions
- Minimum stint length of 8 laps for rivals (except during SC/VSC)
- 5 algorithms: DQN, A2C, TRPO, PPO, RecurrentPPO
- Independent VecNormalize per algorithm for fair comparison
- Configurable GP and agent driver
- Dynamic batch_size and n_steps based on n_laps

Usage:
    python scripts/train_rl_agents.py --gp "Bahrain Grand Prix" --driver Driver_ALO
    python scripts/train_rl_agents.py --gp "Belgian Grand Prix" --driver Driver_VER --algorithms DQN PPO
    python scripts/train_rl_agents.py --gp all --driver Driver_VER
"""

import json
import torch
import argparse
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from sb3_contrib import RecurrentPPO, TRPO

from environment.f1_env_all_drivers import F1EnvAllDrivers, create_f1_env_all_drivers
from hyperparams.config import (
    RECURRENT_PPO_CONFIG, PPO_CONFIG, DQN_CONFIG, A2C_CONFIG, TRPO_CONFIG, TRAINING_CONFIG as HP_CONFIG
)
from model_loader import ModelLoader

# Configure PyTorch for M3 Pro optimization
torch.set_num_threads(8)

# Models output directory
MODELS_DIR = str(config.RL_AGENTS_DIR)

# Training configuration (defaults, can be overridden by command line args)
TRAINING_CONFIG = {
    'n_envs': HP_CONFIG['n_envs'],
    'total_timesteps': HP_CONFIG['total_timesteps'],
    'eval_freq': HP_CONFIG['eval_freq'],
    'n_eval_episodes': HP_CONFIG['n_eval_episodes'],
    'early_stop_patience': HP_CONFIG['early_stop_patience'],
}

# Available Grand Prix
AVAILABLE_GPS = [
    'Bahrain Grand Prix',
    'Saudi Arabian Grand Prix',
    'Miami Grand Prix',
    'Emilia Romagna Grand Prix',
    'Hungarian Grand Prix',
    'Belgian Grand Prix',
    'Dutch Grand Prix',
    'Singapore Grand Prix',
    'United States Grand Prix',
]

# Available drivers
AVAILABLE_DRIVERS = [
    'Driver_ALB', 'Driver_ALO', 'Driver_BOT', 'Driver_GAS', 'Driver_HAM',
    'Driver_HUL', 'Driver_LEC', 'Driver_MAG', 'Driver_NOR', 'Driver_OCO',
    'Driver_PER', 'Driver_PIA', 'Driver_RIC', 'Driver_RUS', 'Driver_SAI',
    'Driver_SAR', 'Driver_STR', 'Driver_TSU', 'Driver_VER', 'Driver_ZHO'
]

NORM_OBS_KEYS = ['tyrelifes', 'positions', 'time_diff_to_agent']

# Available algorithms
AVAILABLE_ALGORITHMS = {
    'DQN': (DQN, DQN_CONFIG),
    'A2C': (A2C, A2C_CONFIG),
    'TRPO': (TRPO, TRPO_CONFIG),
    'PPO': (PPO, PPO_CONFIG),
    'RecurrentPPO': (RecurrentPPO, RECURRENT_PPO_CONFIG),
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RL algorithms for F1 multi-driver environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all algorithms for Bahrain GP with Alonso
  python scripts/train_rl_agents.py --gp "Bahrain Grand Prix" --driver Driver_ALO

  # Train only DQN for Belgian GP with Verstappen
  python scripts/train_rl_agents.py --gp "Belgian Grand Prix" --driver Driver_VER --algorithms DQN

  # Train all algorithms for all GPs with Hamilton
  python scripts/train_rl_agents.py --gp all --driver Driver_HAM

  # Train PPO and RecurrentPPO for Singapore with Norris
  python scripts/train_rl_agents.py --gp "Singapore Grand Prix" --driver Driver_NOR --algorithms PPO RecurrentPPO
        """
    )

    parser.add_argument(
        '--gp',
        type=str,
        required=True,
        help='Grand Prix name (e.g., "Bahrain Grand Prix") or "all" to train on all GPs'
    )

    parser.add_argument(
        '--driver',
        type=str,
        required=True,
        choices=AVAILABLE_DRIVERS,
        help='Agent driver name (e.g., Driver_VER, Driver_HAM, etc.)'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=list(AVAILABLE_ALGORITHMS.keys()),
        default=None,
        help='Algorithms to train. If not specified, trains all algorithms.'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override total timesteps (e.g. 10000 for quick test). Defaults to config value.'
    )

    return parser.parse_args()


def get_initial_positions(gp: str) -> list:
    """Get initial positions for a GP from config file."""
    models_path = str(config.SIMULATION_DIR)
    config_path = os.path.join(models_path, gp, 'config.json')

    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        return cfg.get('initial_positions_list_2024')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def create_env_factory(gp: str, agent_driver: str, loader: ModelLoader,
                       initial_positions: list = None, reward_type: str = 'mix'):
    """Create a factory function for environment creation."""
    def _init():
        return create_f1_env_all_drivers(
            gp=gp,
            driver=agent_driver,
            deterministic=False,
            yf_enabled=True,
            initial_positions=initial_positions,
            loader=loader,
            verbose=False,
            reward_type=reward_type,
        )
    return _init


def create_fresh_envs(env_factory, n_envs: int, normalize_reward: bool = False):
    """Create fresh vectorized environments with independent VecNormalize."""
    vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    eval_vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Reward normalization: required for PPO/RecurrentPPO (huge value losses without it)
    # but disabled for DQN/A2C/TRPO to preserve terminal reward signal
    vec_env = VecNormalize(vec_env, norm_reward=normalize_reward, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
    eval_vec_env = VecNormalize(eval_vec_env, norm_reward=normalize_reward, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)

    return vec_env, eval_vec_env


def create_eval_callback(eval_env, save_path: str, log_path: str,
                         eval_freq: int, n_eval_episodes: int,
                         early_stop_patience: int):
    """Create evaluation callback with early stopping."""
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=early_stop_patience,
        min_evals=10,  # Aumentado de 5 -> 10 para dar mas tiempo inicial
        verbose=1
    )

    return EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        callback_after_eval=stop_callback,
        verbose=0
    )


def train_algorithm(algo_name, algo_cls, algo_config, env_factory, n_envs,
                    output_dir, dynamic_batch_size, dynamic_n_steps,
                    total_timesteps, eval_freq, n_eval_episodes,
                    early_stop_patience):
    """Train a single algorithm with fresh, independent environments."""
    print(f"\n{'─' * 70}")
    print(f"Training {algo_name}")
    print(f"  config: net_arch={algo_config['policy_kwargs'].get('net_arch', 'default')}, "
          f"gamma={algo_config['gamma']}, ent_coef={algo_config.get('ent_coef', 'default')}")
    print(f"{'─' * 70}")

    # Fresh environments for this algorithm
    # Enable reward normalization for PPO/RecurrentPPO (required for value function stability)
    normalize_reward = algo_name.startswith(('PPO', 'RecurrentPPO'))
    vec_env, eval_vec_env = create_fresh_envs(env_factory, n_envs, normalize_reward=normalize_reward)

    if normalize_reward:
        print(f"  Reward normalization: ENABLED (required for {algo_name})")

    algo_dir = os.path.join(output_dir, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    eval_callback = create_eval_callback(
        eval_vec_env,
        save_path=algo_dir,
        log_path=algo_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        early_stop_patience=early_stop_patience
    )

    # Build model kwargs from config
    model_kwargs = {
        'gamma': algo_config['gamma'],
        'policy_kwargs': algo_config['policy_kwargs'],
        'tensorboard_log': os.path.join(output_dir, 'tensorboard'),
        'device': 'cpu',
        'verbose': 1,
    }

    # Add algorithm-specific params
    if algo_name == 'DQN':
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 1e-4),
            'buffer_size': algo_config['buffer_size'],
            'batch_size': algo_config['batch_size'],
            'learning_starts': algo_config.get('learning_starts', 50_000),
            'exploration_fraction': algo_config['exploration_fraction'],
            'exploration_final_eps': algo_config['exploration_final_eps'],
            'target_update_interval': algo_config['target_update_interval'],
            'train_freq': algo_config.get('train_freq', 4),
            'gradient_steps': algo_config.get('gradient_steps', 1),
            'tau': algo_config.get('tau', 1.0),
            'max_grad_norm': algo_config.get('max_grad_norm', 10),
        })
    elif algo_name == 'A2C':
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 7e-4),
            'ent_coef': algo_config['ent_coef'],
            'vf_coef': algo_config.get('vf_coef', 0.5),
            'n_steps': dynamic_n_steps,
        })
    elif algo_name == 'TRPO':
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 1e-3),
            'batch_size': dynamic_batch_size,
            'n_steps': dynamic_n_steps,
            'cg_max_steps': algo_config.get('cg_max_steps', 15),
            'target_kl': algo_config.get('target_kl', 0.01),
        })
    elif algo_name.startswith(('PPO', 'RecurrentPPO')):
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 3e-4),
            'ent_coef': algo_config['ent_coef'],
            'batch_size': dynamic_batch_size,
            'n_steps': dynamic_n_steps,
            'n_epochs': algo_config.get('n_epochs', 10),
            'clip_range': algo_config.get('clip_range', 0.2),
            'vf_coef': algo_config.get('vf_coef', 0.5),
            'max_grad_norm': algo_config.get('max_grad_norm', 0.5),
            'gae_lambda': algo_config.get('gae_lambda', 0.95),
        })

    # Create and train model
    model = algo_cls(algo_config['policy'], vec_env, **model_kwargs)

    model.learn(
        total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name=algo_name
    )

    # Save VecNormalize stats for this algorithm
    vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats.pkl')
    vec_env.save(vecnorm_path)

    # Cleanup
    vec_env.close()
    eval_vec_env.close()

    print(f"  {algo_name} complete. VecNormalize saved to {vecnorm_path}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Determine which algorithms to train
    if args.algorithms:
        algorithms_to_train = args.algorithms
    else:
        # Train all algorithms if none specified
        algorithms_to_train = list(AVAILABLE_ALGORITHMS.keys())

    # Determine which GPs to train on
    if args.gp.lower() == 'all':
        gps_to_train = AVAILABLE_GPS
    else:
        if args.gp not in AVAILABLE_GPS:
            print(f"\nError: '{args.gp}' is not a valid GP.")
            print(f"Available GPs: {', '.join(AVAILABLE_GPS)}")
            sys.exit(1)
        gps_to_train = [args.gp]

    agent_driver = args.driver
    n_envs = TRAINING_CONFIG['n_envs']
    total_timesteps = args.timesteps if args.timesteps is not None else TRAINING_CONFIG['total_timesteps']
    eval_freq = TRAINING_CONFIG['eval_freq']
    n_eval_episodes = TRAINING_CONFIG['n_eval_episodes']
    early_stop_patience = TRAINING_CONFIG['early_stop_patience']

    print("\n" + "=" * 70)
    print("F1 MULTI-DRIVER ENVIRONMENT - RL TRAINING")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  GPs to train: {', '.join(gps_to_train)}")
    print(f"  Agent: {agent_driver}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Eval episodes: {n_eval_episodes}")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  Algorithms to train: {', '.join(algorithms_to_train)}")

    # Initialize model loader (reused across GPs)
    loader = ModelLoader()

    # Train for each GP
    for gp in gps_to_train:
        print("\n" + "=" * 70)
        print(f"LOADING MODELS FOR {gp.upper()}")
        print("=" * 70)

        models = loader.load(gp)
        n_laps = models.n_laps

        print(f"  Laps: {n_laps}")
        print(f"  Compounds: {models.compounds}")
        print(f"  VSC/SC prob: {models.vsc_prob:.2f}/{models.sc_prob:.2f}")

        # Compute dynamic parameters
        dynamic_batch_size = n_laps * n_envs * 2
        dynamic_n_steps = n_laps * 2

        print(f"\nDynamic parameters (n_laps={n_laps}, n_envs={n_envs}):")
        print(f"  batch_size: {dynamic_batch_size}")
        print(f"  n_steps: {dynamic_n_steps}")

        # Get initial positions
        initial_positions = get_initial_positions(gp)
        if initial_positions:
            print(f"  Using 2024 initial positions")
        else:
            print("  No initial positions found, using random")

        # Create output directory
        gp_short = gp.replace(' Grand Prix', '').replace(' ', '_')
        agent_short = agent_driver.replace('Driver_', '')
        output_dir = os.path.join(MODELS_DIR, f"{gp_short}_{agent_short}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output dir: {output_dir}")

        # Environment factory (shared config, each algo gets fresh instances)
        env_factory = create_env_factory(gp, agent_driver, loader, initial_positions)

        print("\n" + "=" * 70)
        print(f"STARTING TRAINING FOR {gp.upper()} (independent VecNormalize per algorithm)")
        print("=" * 70)

        # Shared training args
        train_kwargs = dict(
            env_factory=env_factory,
            n_envs=n_envs,
            output_dir=output_dir,
            dynamic_batch_size=dynamic_batch_size,
            dynamic_n_steps=dynamic_n_steps,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            early_stop_patience=early_stop_patience,
        )

        # Train selected algorithms for this GP
        for algo_name in algorithms_to_train:
            algo_cls, algo_config = AVAILABLE_ALGORITHMS[algo_name]
            train_algorithm(algo_name, algo_cls, algo_config, **train_kwargs)

        # GP Summary
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETED FOR {gp.upper()}")
        print("=" * 70)
        print(f"  Agent: {agent_driver}")
        print(f"  Algorithms trained: {', '.join(algorithms_to_train)}")
        print(f"  Models saved to: {output_dir}")
        print("=" * 70)

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETED")
    print("=" * 70)

    print(f"\nGPs trained: {', '.join(gps_to_train)}")
    print(f"Agent: {agent_driver}")
    print(f"Algorithms trained per GP: {', '.join(algorithms_to_train)}")
    print(f"Total timesteps per algorithm: {total_timesteps:,}")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"  Each GP-algorithm combination has its own vecnormalize_stats.pkl")
    print("=" * 70)
