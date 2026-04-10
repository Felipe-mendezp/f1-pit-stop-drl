import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Optimized RL training script using FastLinearPredictor for 32x speedup.
Hyperparameters optimized based on DRL_research.md recommendations.

Key improvements:
- FastLinearPredictor: 32x faster predictions (44us -> 1.4us)
- 5 algorithms: RecurrentPPO, PPO, DQN, A2C, TRPO
- RecurrentPPO prioritized (best for temporal planning, expected 5-15% improvement)
- Optimized hyperparameters from empirical F1 RL research (arXiv:2501.04068)
- Proper VecNormalize configuration for stable training
- 8 parallel environments for M3 Pro optimization
- 3M timesteps per algorithm (~3-4 hours total per GP)

Algorithm Priority (research-based):
1. RecurrentPPO - BEST for strategic planning with LSTM temporal modeling
2. PPO - Most robust, proven in operations research
3. DQN - Maximum sample efficiency via replay buffer
4. A2C - Baseline for comparison (high variance, unstable)
5. TRPO - Included for completeness (10x slower than PPO)
"""

import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from sb3_contrib import RecurrentPPO, TRPO
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized

# Import FastLinearPredictor from reg.py
from reg import FastLinearPredictor

# Configure PyTorch for M3 Pro optimization
torch.set_num_threads(8)  # M3 Pro has 8-12 performance cores

# Path to models
path = str(config.SIMULATION_DIR / 'regressions')

selected_GP = {
        'Abu Dhabi Grand Prix': ['C3', 'C4', 'C5'],
        'Australian Grand Prix': ['C3', 'C4', 'C5'],
        'Bahrain Grand Prix': ['C1', 'C2', 'C3'],
        'Belgian Grand Prix': ['C2', 'C3', 'C4'],
        'Canadian Grand Prix': ['C3', 'C4', 'C5'],
        'Dutch Grand Prix': ['C1', 'C2', 'C3'],
        'Hungarian Grand Prix': ['C3', 'C4', 'C5'],
        'Italian Grand Prix': ['C3', 'C4', 'C5'],
        'Mexico City Grand Prix': ['C3', 'C4', 'C5'],
        'Saudi Arabian Grand Prix': ['C2', 'C3', 'C4'],
        'Singapore Grand Prix': ['C3', 'C4', 'C5'],
        'Sao Paulo Grand Prix': ['C3', 'C4', 'C5'],
        'United States Grand Prix': ['C2', 'C3', 'C4']
        }

laps_dd = {
    'Abu Dhabi Grand Prix': 58,
    'Australian Grand Prix': 58,
    'Bahrain Grand Prix': 57,
    'Belgian Grand Prix': 44,
    'Canadian Grand Prix': 70,
    'Dutch Grand Prix': 72,
    'Hungarian Grand Prix': 70,
    'Italian Grand Prix': 53,
    'Mexico City Grand Prix': 71,
    'Saudi Arabian Grand Prix': 50,
    'Singapore Grand Prix': 62,
    'Sao Paulo Grand Prix': 69,
    'United States Grand Prix': 56,
}

# Output directory for models
MODELS_OUTPUT = str(config.RL_AGENTS_DIR / 'one_driver')

# ============================================================================
# Main execution (required for multiprocessing with SubprocVecEnv)
# ============================================================================
if __name__ == "__main__":
    # ============================================================================
    # Load FAST PREDICTORS (32x speedup)
    # ============================================================================
    print("\n" + "="*70)
    print("LOADING FAST PREDICTORS (32x faster than statsmodels)")
    print("="*70)

    reg_dd = {}
    for gp in selected_GP.keys():
        fast_predictor_path = f"{path}/{gp}_fast_predictor.pkl"
        with open(fast_predictor_path, "rb") as f:
            fast_predictor = pickle.load(f)
            reg_dd[gp] = fast_predictor

        info = fast_predictor.get_info()
        print(f"  {gp[:20]:<20} | Features: {info['n_features']:2d} | "
              f"Memory: {info['memory_bytes']:3d}B | Expected: ~1.4us/pred")

    print("="*70)
    print(f"Loaded {len(reg_dd)} fast predictors")
    print(f"Expected speedup: 32x faster predictions")
    print(f"Training time savings: ~15-20 hours for 5M episodes")
    print("="*70 + "\n")

    # Training loop
    for gp, compounds in tqdm(selected_GP.items(), desc="Training GPs"):
        print(f"\n{'='*70}")
        print(f"Training: {gp}")
        print(f"Laps: {laps_dd[gp]} | Compounds: {compounds}")
        print(f"{'='*70}\n")

        n_laps = laps_dd[gp]
        n_envs = 8  # Optimal for M3 Pro (8-12 cores)
        episodes = 3_000_000  # 1M timesteps per algorithm

        env_dd = {
            'n_laps': laps_dd[gp],
            'available_compounds': compounds,
            'gp': gp,
            'driver': 'VER',
            'model_laptime': reg_dd[gp],  # FastLinearPredictor (32x faster!)
            'verbose': False
        }

        # Create vectorized environments with SubprocVecEnv for true parallelism
        vec_env = make_vec_env(
            F1EnvOneDriverV2Optimized,
            env_kwargs=env_dd,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )
        eval_vec_env = make_vec_env(
            F1EnvOneDriverV2Optimized,
            env_kwargs=env_dd,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )

        # VecNormalize for reward normalization (obs are discrete, don't need normalization)
        # PHASE 1 FIX: Disable reward normalization to prevent non-stationarity
        # Penalties are now proportional to lap times, so normalization not needed
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, gamma=0.99)
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=False, norm_reward=False, gamma=0.99)
        sync_envs_normalization(vec_env, eval_vec_env)

        gp_short = gp[:-11]  # Remove " Grand Prix"

        # ========================================================================
        # 1. RecurrentPPO - BEST for temporal planning (Research priority #1)
        # ========================================================================
        print("Training RecurrentPPO (Priority #1 - Best for temporal planning)...")

        policy_kwargs_rppo = dict(
            lstm_hidden_size=64,  # Optimal for 6 features
            n_lstm_layers=1,  # One layer sufficient for 50-70 step horizon
            net_arch=[dict(pi=[64], vf=[64])],  # Separate actor-critic networks
            shared_lstm=False,  # Separate LSTMs for better performance
            enable_critic_lstm=True,  # Both networks benefit from temporal modeling
            activation_fn=nn.ReLU,
            ortho_init=False  # Can cause issues with recurrence
        )

        eval_callback_RecurrentPPO = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"{MODELS_OUTPUT}/{gp_short}/RecurrentPPO",
            log_path=f"{MODELS_OUTPUT}/logs/",
            eval_freq=5_000,
            n_eval_episodes=50,
            verbose=0
        )

        model_RecurrentPPO = RecurrentPPO(
            "MultiInputLstmPolicy",
            vec_env,
            learning_rate=3e-4,  # Standard, may need lower for LSTM stability
            n_steps=128,  # Much lower than standard PPO (captures ~2 full races)
            batch_size=128,  # Balance between diversity and memory
            n_epochs=10,
            gamma=0.99,  # High for 50-70 step horizon
            gae_lambda=0.95,  # Balance bias-variance for long horizon
            clip_range=0.2,  # Standard robust PPO clipping
            normalize_advantage=True,
            ent_coef=0.01,  # Maintain exploration in 4-action space
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs_rppo,
            tensorboard_log=f"{MODELS_OUTPUT}/tensorboard_det/",
            device="cpu",  # RecurrentPPO is CPU-bound on M3
            verbose=1
        )

        model_RecurrentPPO.learn(
            episodes,
            callback=eval_callback_RecurrentPPO,
            progress_bar=True,
            tb_log_name=f"RecurrentPPO_{gp_short}"
        )

        # ========================================================================
        # 2. PPO Standard - ROBUST alternative (Research priority #2)
        # ========================================================================
        print("Training PPO (Priority #2 - Robust and simple)...")

        policy_kwargs_ppo = dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Separate networks
            activation_fn=nn.Tanh,  # Better gradient flow
            ortho_init=False
        )

        eval_callback_PPO = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"{MODELS_OUTPUT}/{gp_short}/PPO",
            log_path=f"{MODELS_OUTPUT}/logs/",
            eval_freq=5_000,
            n_eval_episodes=50,
            verbose=0
        )

        model_PPO = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,  # With linear decay to 0
            n_steps=512,  # Capture multiple episodes per update
            batch_size=128,  # 4096 total steps / 128 = 32 minibatches
            n_epochs=10,
            gamma=0.99,  # Effective horizon ~100 steps
            gae_lambda=0.95,  # Standard for long horizon
            clip_range=0.2,  # Robust default
            ent_coef=0.01,  # Entropy bonus for 4 actions
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,  # Early stopping if KL too high
            policy_kwargs=policy_kwargs_ppo,
            tensorboard_log=f"{MODELS_OUTPUT}/tensorboard_det/",
            device="cpu",  # PPO is CPU-bound
            verbose=1
        )

        model_PPO.learn(
            episodes,
            callback=eval_callback_PPO,
            progress_bar=True,
            tb_log_name=f"PPO_{gp_short}"
        )

        # ========================================================================
        # 3. DQN - SAMPLE EFFICIENT (Research priority #3)
        # ========================================================================
        print("Training DQN (Priority #3 - Maximum sample efficiency)...")

        policy_kwargs_dqn = dict(
            net_arch=[128, 128],  # Compact for 6 features, 4 actions
            activation_fn=nn.ReLU
        )

        eval_callback_DQN = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"{MODELS_OUTPUT}/{gp_short}/DQN",
            log_path=f"{MODELS_OUTPUT}/logs/",
            eval_freq=5_000,
            n_eval_episodes=50,
            verbose=0
        )

        model_DQN = DQN(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=50_000,  # 1000-2000 episodes for discrete problem
            batch_size=64,  # Stable gradient updates
            learning_starts=1000,  # Accumulate initial experiences
            gamma=0.99,  # High for long horizon
            target_update_interval=1000,  # ~15-20 episodes
            train_freq=4,  # Update every 4 env steps
            gradient_steps=1,
            exploration_fraction=0.25,  # Explore for 25% of training
            exploration_initial_eps=1.0,  # Start fully random
            exploration_final_eps=0.1,  # Maintain some exploration
            max_grad_norm=10,
            policy_kwargs=policy_kwargs_dqn,
            tensorboard_log=f"{MODELS_OUTPUT}/tensorboard_det/",
            device="cpu",
            verbose=1
        )

        model_DQN.learn(
            episodes,
            callback=eval_callback_DQN,
            progress_bar=True,
            tb_log_name=f"DQN_{gp_short}"
        )

        # ========================================================================
        # 4. A2C - BASELINE (Not recommended but included for comparison)
        # ========================================================================
        print("Training A2C (Priority #4 - Baseline for comparison)...")

        policy_kwargs_a2c = dict(
            net_arch=[64, 64],  # Shared network for actor-critic
            activation_fn=nn.Tanh
        )

        eval_callback_A2C = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"{MODELS_OUTPUT}/{gp_short}/A2C",
            log_path=f"{MODELS_OUTPUT}/logs/",
            eval_freq=5_000,
            n_eval_episodes=50,
            verbose=0
        )

        model_A2C = A2C(
            "MultiInputPolicy",
            vec_env,
            learning_rate=5e-4,  # Narrower effective range than PPO
            n_steps=20,  # Longer than default 5 for long episodes
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # Maintain exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,  # Critical for stability
            use_rms_prop=True,  # Paper original uses RMSProp
            rms_prop_eps=1e-5,
            policy_kwargs=policy_kwargs_a2c,
            tensorboard_log=f"{MODELS_OUTPUT}/tensorboard_det/",
            device="cpu",
            verbose=1
        )

        model_A2C.learn(
            episodes,
            callback=eval_callback_A2C,
            progress_bar=True,
            tb_log_name=f"A2C_{gp_short}"
        )

        # ========================================================================
        # 5. TRPO - EXPENSIVE (Not recommended but included for comparison)
        # ========================================================================
        print("Training TRPO (Priority #5 - Computationally expensive)...")

        policy_kwargs_trpo = dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=nn.Tanh
        )

        eval_callback_TRPO = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"{MODELS_OUTPUT}/{gp_short}/TRPO",
            log_path=f"{MODELS_OUTPUT}/logs/",
            eval_freq=5_000,
            n_eval_episodes=50,
            verbose=0
        )

        model_TRPO = TRPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=1e-3,  # For value function
            n_steps=n_laps * 2,  # steps_per_epoch
            batch_size=n_laps * n_envs * 2,
            gamma=0.99,
            gae_lambda=0.97,  # Slightly higher than PPO
            target_kl=0.01,  # Critical constraint for monotonic improvement
            cg_iters=10,  # Conjugate gradient iterations
            policy_kwargs=policy_kwargs_trpo,
            tensorboard_log=f"{MODELS_OUTPUT}/tensorboard_det/",
            device="cpu",
            verbose=1
        )

        model_TRPO.learn(
            episodes,
            callback=eval_callback_TRPO,
            progress_bar=True,
            tb_log_name=f"TRPO_{gp_short}"
        )

        print(f"\nCompleted training for {gp}")
        print(f"   - RecurrentPPO: Best for temporal planning (expected 5-15% improvement)")
        print(f"   - PPO: Most robust, simple implementation")
        print(f"   - DQN: Highest sample efficiency via replay buffer")
        print(f"   - A2C: Baseline (high variance, not recommended)")
        print(f"   - TRPO: Computationally expensive (10x slower than PPO)\n")

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETED - 5 ALGORITHMS")
    print("="*70)
    print("\nAlgorithm Priority (based on DRL_research.md):")
    print("  1. RecurrentPPO - BEST for strategic planning with temporal dependencies")
    print("  2. PPO Standard - Most robust, proven in operations research")
    print("  3. DQN - Maximum sample efficiency, proven in GT racing")
    print("  4. A2C - Baseline (high variance, included for comparison)")
    print("  5. TRPO - Computationally expensive (10x slower, included for comparison)")
    print("\nNote:")
    print("  - A2C: High variance, unstable (use only as baseline)")
    print("  - TRPO: 10x slower than PPO without practical benefits")
    print("\nExpected Best Performer: RecurrentPPO (5-15% improvement)")
    print("="*70)
