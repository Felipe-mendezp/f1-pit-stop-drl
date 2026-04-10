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
    'n_envs': 10,                    # Menos envs → más episodios por env (~8600)
    'eval_freq': 15_000,             # Evaluación más frecuente para early stopping preciso
    'n_eval_episodes': 50,           # Reducido para ahorrar tiempo de evaluación
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
# DQN_CONFIG (previous - median P8, IQR P5-P11)
# DQN_CONFIG = {
#     'policy': 'MultiInputPolicy',
#     'gamma': 0.995,
#     'buffer_size': 1_500_000,
#     'batch_size': 256,
#     'exploration_fraction': 0.3,
#     'exploration_final_eps': 0.03,
#     'target_update_interval': 1500,
#     'policy_kwargs': {
#         'activation_fn': nn.ReLU,
#         'net_arch': [256, 256, 256],
#     },
# }

# DQN_CONFIG v2 (mean P10.18, IQR 5.37 — worst among 5 algos)
# DQN_CONFIG = {
#     'policy': 'MultiInputPolicy',
#     'gamma': 0.995,
#     'learning_rate': 5e-4,
#     'buffer_size': 1_500_000,
#     'batch_size': 512,
#     'learning_starts': 10_000,
#     'exploration_fraction': 0.4,
#     'exploration_final_eps': 0.05,
#     'target_update_interval': 1000,
#     'policy_kwargs': {
#         'activation_fn': nn.ReLU,
#         'net_arch': [256, 128],
#     },
# }

# DQN_CONFIG v3 (mean P11.87 — too many changes at once)
# DQN_CONFIG = {
#     'policy': 'MultiInputPolicy',
#     'gamma': 0.99, 'learning_rate': linear_schedule(5e-4, 5e-5),
#     'buffer_size': 500_000, 'batch_size': 256, 'learning_starts': 10_000,
#     'train_freq': 4, 'gradient_steps': 4, 'tau': 0.01,
#     'target_update_interval': 500, 'exploration_fraction': 0.5,
#     'exploration_final_eps': 0.08, 'max_grad_norm': 1.0,
#     'policy_kwargs': {'activation_fn': nn.ReLU, 'net_arch': [512, 256, 128]},
# }

# DQN_CONFIG v4 (mean P10.56 — LR schedule + bigger net on v2 base)
# DQN_CONFIG = {
#     'policy': 'MultiInputPolicy',
#     'gamma': 0.995, 'learning_rate': linear_schedule(5e-4, 5e-5),
#     'buffer_size': 1_500_000, 'batch_size': 512, 'learning_starts': 10_000,
#     'exploration_fraction': 0.4, 'exploration_final_eps': 0.05,
#     'target_update_interval': 1000,
#     'policy_kwargs': {'activation_fn': nn.ReLU, 'net_arch': [512, 256, 128]},
# }

# v5: Based on v1 (median P8) + LR schedule as only new addition
DQN_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': linear_schedule(3e-4, 1e-5),           # v1 used 1e-4 fixed; start 3x faster, decay to avoid late oscillation.
    'buffer_size': 1_500_000,
    'batch_size': 256,                                      # v1's value (smaller → more diverse gradient updates).
    'learning_starts': 30_000,                              # Compromise: v1=50K, v2=10K. Let buffer fill before learning.
    'exploration_fraction': 0.3,                            # v1's value.
    'exploration_final_eps': 0.05,                          # Slight bump from v1's 0.03 for non-stationary env.
    'target_update_interval': 1500,                         # v1's value (slower → more stable target network).
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 256, 256],                        # v1's architecture (3-layer, more capacity for 203-259 dim input).
    },
}

# =============================================================================
# 2. A2C
# =============================================================================
# n_steps is set dynamically in train script (= n_laps * 2)
A2C_CONFIG = {
    'policy': 'MultiInputPolicy',
    'gamma': 0.995,
    'learning_rate': 5e-4,   # Más alto para convergencia rápida con 5M
    'ent_coef': 0.01,        # Menos ruido exploratorio → explotación más rápida
    'vf_coef': 0.5,
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'net_arch': [256, 128],  # Más compacta: suficiente para Discrete(4)
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
    'target_kl': 0.02,      # Más permisivo → pasos de policy más grandes con pocos datos
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
    'policy_kwargs': {
        'activation_fn': nn.ReLU,
        'lstm_hidden_size': 64,
        'net_arch': [128],
        'n_lstm_layers': 1,
        'enable_critic_lstm': True,
    },
}
