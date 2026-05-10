"""
Hyperparameters for F1 RL Training

Only explicitly set parameters are included per algorithm.
All other parameters use SB3 defaults.

Dynamic parameters (computed at runtime based on n_laps and n_envs):
- batch_size = n_laps * n_envs * 2
- n_steps = n_laps * 2

Environment Characteristics:
- Episode length: ~58 steps (57 laps + compound selection)
- Observation space: Dict with ~123 dimensions
- Action space: Discrete(4) - no pit, pit to compound 1/2/3
"""

import torch.nn as nn

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAINING_CONFIG = {
    'total_timesteps': 5_000_000,
    'n_envs': 10,                    # Fewer envs -> more episodes per env (~8600)
    'eval_freq': 15_000,             # More frequent evaluation for precise early stopping
    'n_eval_episodes': 50,           # Reduced to save evaluation time
    'early_stop_patience': 25,
}

# =============================================================================
# Learning rate schedule: linear decay from lr_start to lr_end
# =============================================================================
def linear_schedule(lr_start: float, lr_end: float):
    """Returns a callable that computes lr = lr_start - (lr_start - lr_end) * progress."""
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        return lr_end + (lr_start - lr_end) * progress_remaining
    return _schedule

# =============================================================================
# 1. DQN
# =============================================================================
DQN_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': linear_schedule(3e-4, 1e-5),  # Annealed: start fast, decay to avoid late oscillation
    'buffer_size': 1_500_000,
    'batch_size': 256,
    'learning_starts': 30_000,                     # Let buffer accumulate before learning
    'exploration_fraction': 0.3,
    'exploration_final_eps': 0.05,
    'target_update_interval': 1500,                # Slower target updates -> more stable
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 256, 256],               # 3-layer net for the 203-259 dim input
    },
}

# =============================================================================
# 2. A2C
# =============================================================================
# n_steps is set dynamically in train script (= n_laps * 2)
A2C_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': 5e-4,   # Higher LR for faster convergence within 5M timesteps
    'ent_coef': 0.01,        # Lower exploration noise -> faster exploitation
    'vf_coef': 0.5,
    'gae_lambda': 1.0,       # Matches Table F.2 (high-variance Monte-Carlo advantage)
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 128],  # Compact: sufficient for Discrete(4) action space
    },
}

# =============================================================================
# 3. TRPO
# =============================================================================
TRPO_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': 5e-4,
    'cg_max_steps': 15,
    'target_kl': 0.02,       # Looser KL bound -> larger policy steps when data is scarce
    'gae_lambda': 0.95,
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 128],
    },
}

# =============================================================================
# 4. PPO
# =============================================================================
PPO_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': 3e-4,          # Reduced from 5e-4: slower, more stable learning
    'ent_coef': 0.05,               # Increased from 0.02: 2.5x more exploration bonus
    'n_epochs': 3,                  # Reduced from 4: less overfitting per batch
    'clip_range': 0.2,              # Reduced from 0.25: more conservative updates
    'vf_coef': 0.01,                # Keep: worked well with reward normalization
    'max_grad_norm': 0.5,           # Reduced from 0.6: prevent large parameter jumps
    'gae_lambda': 0.98,             # Add: higher lambda for better long-term credit assignment
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 128],
    },
}

# =============================================================================
# 5. RECURRENT PPO
# =============================================================================
RECURRENT_PPO_CONFIG = {
    'policy': 'MultiInputLstmPolicy',
    'gamma': 0.995,
    'learning_rate': 5e-4,
    'ent_coef': 0.02,
    'n_epochs': 4,
    'clip_range': 0.25,
    'vf_coef': 0.01,
    'max_grad_norm': 0.6,
    'gae_lambda': 0.95,
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'lstm_hidden_size': 64,
        'net_arch': [128],
        'n_lstm_layers': 1,
        'enable_critic_lstm': True,
    },
}
