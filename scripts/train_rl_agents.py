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
import random
import numpy as np
import torch
import argparse
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback, CheckpointCallback
)
from sb3_contrib import RecurrentPPO, TRPO

from double_dqn import DoubleDQN

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

# Available algorithms. 'DQN' uses DoubleDQN (double-DQN TD target) so the
# implementation matches the "Double DQN" reported in the paper; the
# checkpoints remain loadable with stable_baselines3.DQN.
AVAILABLE_ALGORITHMS = {
    'DQN': (DoubleDQN, DQN_CONFIG),
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

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility (numpy, torch, SB3, env). Default: 42.'
    )

    parser.add_argument(
        '--hyperparams',
        default='v1',
        help='Hyperparameter set: v1 = paper protocol (default); v2 = '
             'hyperparams/config_v2.py; any other NAME resolves to '
             'hyperparams/config_NAME.py and overrides RecurrentPPO only, so '
             'successive tuning iterations do not need a code change.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override the models output directory (default: trained_models/rl_agents). '
             'Use e.g. trained_models/rl_agents_v2 to keep the paper models intact.'
    )

    # ---- Speed knobs (opt-in; defaults reproduce the paper protocol) ----
    parser.add_argument(
        '--fast-env',
        action='store_true',
        help='Enable the numpy fast path for rival/overtake logits (~3x faster env '
             'steps; numerically validated against the statsmodels path at init).'
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=None,
        help='Override number of parallel envs (default: hyperparams config, 10). '
             'Use 5-6 when running two GPs concurrently.'
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=None,
        help='Override torch.set_num_threads (default 8). Use 2-4 to free cores '
             'for the env workers.'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=None,
        help='Override eval frequency in VEC steps (default: config, 15000 = every '
             '150k transitions with 10 envs). 5000 makes early stopping operative.'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=None,
        help='Override early-stopping patience in evaluations (default: config, 25).'
    )

    # ---- Deep research 2026-08-09 (P3/P4; opt-in, default off = v1 intacto) ----
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=0,
        help='P3: guardar checkpoints intermedios cada N vec-steps (con '
             'save_vecnormalize=True, a <algo_dir>/ck/). 0 = off (default). '
             'Con 10 envs, 10000 vec-steps = 100k transiciones = ~50 ck a 5M.'
    )
    parser.add_argument(
        '--ent-target',
        type=float,
        default=None,
        help='P4: controlador de entropía en lazo cerrado hacia este objetivo '
             'en nats (p.ej. 0.15). Sustituye al anneal de lazo abierto si la '
             'config lo tuviera. None = off (default).'
    )
    parser.add_argument(
        '--ent-target-k',
        type=float,
        default=0.5,
        help='P4: ganancia del controlador de entropía (default 0.5).'
    )
    parser.add_argument(
        '--norm-obs-off',
        action='store_true',
        help='P5: entrenar con observaciones CRUDAS (norm_obs_keys=[]; el '
             'z-scoring por slot destruye el orden de pista). norm_reward '
             'se mantiene. Default off = protocolo v1.'
    )
    parser.add_argument(
        '--sc-bunching',
        action='store_true',
        help='Mecánica SC del autor (2026-08-28): pit solo en la 1a vuelta '
             'de SC, coef SC plano, y aglomeración del pelotón detrás del '
             'líder desde la 3a vuelta de SC (gaps borrados). Default off.'
    )
    parser.add_argument(
        '--norm-reward-off',
        action='store_true',
        help='Investigación Miami (2026-09-02): desactivar VecNormalize de '
             'reward en PPO/RecurrentPPO (como DQN/A2C/TRPO). Default off.'
    )
    parser.add_argument(
        '--rule-mask',
        action='store_true',
        help='Máscara de reglas en entrenamiento (2026-09-01): el entorno '
             'hace cumplir la regla de dos compuestos (pit al mismo compuesto '
             'sin cumplirla -> no-op; no parar en la última vuelta factible -> '
             'parada forzada legal). Misma lógica que --legal-mask del decision '
             'support. Default off.'
    )
    parser.add_argument(
        '--sc-bunch-delta',
        type=float,
        default=1.5,
        help='Espaciado de la fila bajo SC en segundos (calibrado con datos '
             'reales: mediana 1.56s). Default 1.5.'
    )

    # ---- Env-physics fixes (2026-08-04/05; opt-in, default = legacy physics
    # so existing agent sets remain reproducible). Same semantics as in
    # decision_support_rollouts.py; see the diagnosis comment block next to
    # PIT_PASS_DETERMINISTIC_DEFAULT in f1_env_all_drivers.py. ----
    parser.add_argument(
        '--pit-det-pass',
        action='store_true',
        help='Fix (a): battles involving a driver on an in-/out-lap resolve '
             'deterministically (no pit-lane weld / free pit stop).'
    )
    parser.add_argument(
        '--following-gap-scale',
        type=float,
        default=0.0,
        help='Fix (b): denied overtakes settle at 0.4 + Exp(scale) seconds '
             'instead of the constant 0.4s. Calibrated value: 1.0. 0.0 = legacy.'
    )

    return parser.parse_args()


def seed_all(seed: int) -> None:
    """Seed every RNG that affects training reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                       initial_positions: list = None, reward_type: str = 'mix',
                       fast_rivals: bool = False,
                       pit_det_pass: bool = False,
                       following_gap_scale: float = 0.0,
                       sc_bunching: bool = False,
                       sc_bunch_delta: float = 1.5,
                       rule_mask: bool = False):
    """Create a factory function for environment creation."""
    def _init():
        env = create_f1_env_all_drivers(
            gp=gp,
            driver=agent_driver,
            deterministic=False,
            yf_enabled=True,
            initial_positions=initial_positions,
            loader=loader,
            verbose=False,
            reward_type=reward_type,
            fast_rivals=fast_rivals,
        )
        # Opt-in physics fixes; set inside _init so every SubprocVecEnv worker
        # (train AND eval env) gets them. Default keeps legacy physics.
        env.pit_pass_deterministic = pit_det_pass
        env.following_gap_scale = following_gap_scale
        env.sc_bunching = sc_bunching
        env.sc_bunch_delta = sc_bunch_delta
        env.rule_mask = rule_mask
        return env
    return _init


def create_fresh_envs(env_factory, n_envs: int, normalize_reward: bool = False,
                      gamma: float = 0.995, seed: int = None,
                      norm_obs_keys=None):
    """Create fresh vectorized environments with independent VecNormalize.

    norm_obs_keys=None -> NORM_OBS_KEYS (protocolo v1). P5 (deep research
    2026-08-09) pasa [] via --norm-obs-off: obs crudas (el z-scoring por slot
    destruye el orden de pista; VecNormalize sigue con norm_reward para PPO).
    """
    if norm_obs_keys is None:
        norm_obs_keys = NORM_OBS_KEYS
    vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed)
    eval_seed = None if seed is None else seed + 10_000
    eval_vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=eval_seed)

    # Reward normalization: required for PPO/RecurrentPPO (huge value losses without it)
    # but disabled for DQN/A2C/TRPO to preserve terminal reward signal.
    # gamma must match the agent's discount (return normalization uses it).
    vec_env = VecNormalize(vec_env, norm_reward=normalize_reward,
                           norm_obs_keys=norm_obs_keys, clip_reward=10000.0,
                           gamma=gamma)
    # The eval env NEVER updates its own statistics (training=False): it
    # receives the train env's stats via sync_envs_normalization before
    # each evaluation, so the model is evaluated with the exact
    # normalization it was trained with. Rewards are kept raw.
    eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_reward=False,
                                norm_obs_keys=norm_obs_keys, clip_reward=10000.0,
                                gamma=gamma)

    return vec_env, eval_vec_env


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal `model.ent_coef` from `start` to `final` over training.

    SB3 takes ent_coef as a plain float (`self.ent_coef * entropy_loss`), so a
    schedule cannot be passed to the constructor the way learning_rate or
    clip_range can; it has to be written on the model as training progresses.

    Motivation in hyperparams/config_tuned.py: with a constant 0.02 the policy
    entropy falls to ~1.5% of its maximum inside the first 10% of the budget and
    the run freezes there.
    """

    def __init__(self, start: float, final: float, total_timesteps: int,
                 anneal_frac: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.final = final
        self.total = max(1, int(total_timesteps))
        # Reach `final` after this fraction of training and hold it. Annealing to
        # ~0 early leaves a tail of near-greedy optimization, which matters
        # because the policy is DEPLOYED with argmax: the wider the train/deploy
        # stochasticity gap at the end, the worse the greedy policy can be (see
        # config_tuned's 6-stop failure).
        self.anneal_frac = max(1e-6, float(anneal_frac))

    def _on_training_start(self) -> None:
        self.model.ent_coef = self.start

    def _on_step(self) -> bool:
        progress = min(1.0, (self.num_timesteps / self.total) / self.anneal_frac)
        self.model.ent_coef = self.start + (self.final - self.start) * progress
        if self.verbose and self.num_timesteps % 500_000 < self.model.n_envs:
            print(f"  [entropy anneal] step {self.num_timesteps}: "
                  f"ent_coef={self.model.ent_coef:.4f}")
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('train/ent_coef', float(self.model.ent_coef))


class EntropyTargetCallback(BaseCallback):
    """P4 (deep research 2026-08-09): controlador multiplicativo en LAZO CERRADO
    sobre la entropía ALCANZADA, una actualización por rollout.

    A diferencia de EntropyAnnealCallback (lazo abierto sobre el COEFICIENTE),
    este lee la entropía lograda del último update (train/entropy_loss, ya
    enmascarada por sb3-contrib) y ajusta ent_coef para perseguir un objetivo:

        ent_coef <- clip(ent_coef * exp(k * (H_target(t) - H)), 1e-4, 0.30)

    H_target(t) = h_target hasta hold_frac del presupuesto, luego lineal a
    h_final en anneal_end_frac, y sostenido. Telemetría por rollout en
    entctl/* (ent_coef, H, H_target) para el kill-test del piloto.
    NUNCA instalar junto a EntropyAnnealCallback: pelearían por model.ent_coef.
    """

    def __init__(self, total_timesteps: int, h_target: float = 0.15,
                 h_final: float = 0.03, hold_frac: float = 0.7,
                 anneal_end_frac: float = 0.9, k: float = 0.5,
                 coef_min: float = 1e-4, coef_max: float = 0.30,
                 csv_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.total = max(1, int(total_timesteps))
        self.h_target = float(h_target)
        self.h_final = float(h_final)
        self.hold_frac = float(hold_frac)
        self.anneal_end_frac = max(float(anneal_end_frac), self.hold_frac + 1e-6)
        self.k = float(k)
        self.coef_min = float(coef_min)
        self.coef_max = float(coef_max)
        # Telemetría plana para el kill-test G1 (el stdout de SB3 trunca los
        # nombres de columna y el TB requiere parsear event files).
        self.csv_path = csv_path
        if csv_path and not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('num_timesteps,ent_coef,H,H_target\n')

    def _target(self) -> float:
        prog = self.num_timesteps / self.total
        if prog <= self.hold_frac:
            return self.h_target
        if prog >= self.anneal_end_frac:
            return self.h_final
        f = (prog - self.hold_frac) / (self.anneal_end_frac - self.hold_frac)
        return self.h_target + (self.h_final - self.h_target) * f

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        target = self._target()
        # logger.name_to_value es defaultdict(float): comprobar pertenencia
        # ANTES de leer, o el primer rollout usaría H=0 espurio.
        h = None
        if 'train/entropy_loss' in self.logger.name_to_value:
            h = -float(self.logger.name_to_value['train/entropy_loss'])
            self.model.ent_coef = float(np.clip(
                self.model.ent_coef * np.exp(self.k * (target - h)),
                self.coef_min, self.coef_max))
            self.logger.record('entctl/H', h)
        self.logger.record('entctl/ent_coef', float(self.model.ent_coef))
        self.logger.record('entctl/H_target', target)
        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                f.write(f"{self.num_timesteps},{self.model.ent_coef:.6f},"
                        f"{'' if h is None else round(h, 6)},{target:.4f}\n")


class SyncedEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize stats train -> eval before each evaluation."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


class SaveVecNormalizeOnBest(BaseCallback):
    """
    Save the TRAIN env's VecNormalize statistics whenever a new best model
    is saved, so vecnormalize_stats.pkl always matches best_model.zip
    (instead of the stats at the END of training, which may correspond to a
    much later point than the checkpoint).
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(os.path.join(self.save_path, 'vecnormalize_stats.pkl'))
        return True


def create_eval_callback(eval_env, save_path: str, log_path: str,
                         eval_freq: int, n_eval_episodes: int,
                         early_stop_patience: int):
    """Create evaluation callback with early stopping and stat syncing."""
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=early_stop_patience,
        min_evals=10,  # Raised from 5 to 10 to give early training more time before being stopped
        verbose=1
    )

    return SyncedEvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        callback_after_eval=stop_callback,
        callback_on_new_best=SaveVecNormalizeOnBest(save_path),
        verbose=0
    )


def train_algorithm(algo_name, algo_cls, algo_config, env_factory, n_envs,
                    output_dir, dynamic_batch_size, dynamic_n_steps,
                    total_timesteps, eval_freq, n_eval_episodes,
                    early_stop_patience, seed: int = 42,
                    checkpoint_freq: int = 0, ent_target: float = None,
                    ent_target_k: float = 0.5, norm_obs_off: bool = False,
                    norm_reward_off: bool = False):
    """Train a single algorithm with fresh, independent environments."""
    print(f"\n{'─' * 70}")
    print(f"Training {algo_name}")
    print(f"  config: net_arch={algo_config['policy_kwargs'].get('net_arch', 'default')}, "
          f"gamma={algo_config['gamma']}, ent_coef={algo_config.get('ent_coef', 'default')}")
    print(f"{'─' * 70}")

    # Fresh environments for this algorithm
    # Enable reward normalization for PPO/RecurrentPPO (required for value function stability)
    normalize_reward = algo_name.startswith(('PPO', 'RecurrentPPO')) and not norm_reward_off
    vec_env, eval_vec_env = create_fresh_envs(
        env_factory, n_envs, normalize_reward=normalize_reward,
        gamma=algo_config['gamma'], seed=seed,
        norm_obs_keys=([] if norm_obs_off else None),
    )

    if normalize_reward:
        print(f"  Reward normalization: ENABLED (required for {algo_name})")
    if norm_obs_off:
        print("  Obs normalization: OFF (P5, norm_obs_keys=[])")

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
        'seed': seed,
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
        # V2: n-step replay targets (SB3 >= 2.7)
        if 'n_steps' in algo_config:
            model_kwargs['n_steps'] = algo_config['n_steps']
    elif algo_name == 'A2C':
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 7e-4),
            'ent_coef': algo_config['ent_coef'],
            'vf_coef': algo_config.get('vf_coef', 0.5),
            'gae_lambda': algo_config.get('gae_lambda', 1.0),
            'n_steps': dynamic_n_steps,
        })
    elif algo_name == 'TRPO':
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 1e-3),
            'batch_size': dynamic_batch_size,
            'n_steps': dynamic_n_steps,
            'cg_max_steps': algo_config.get('cg_max_steps', 15),
            'target_kl': algo_config.get('target_kl', 0.01),
            'gae_lambda': algo_config.get('gae_lambda', 0.95),
        })
    elif algo_name.startswith(('PPO', 'RecurrentPPO')):
        # P2 (deep research 2026-08-09, hyperparams/config_align.py): alinear el
        # rollout con la longitud REAL del episodio (n_laps+1 pasos). Config-driven
        # y por-algoritmo: default 0 en v1/v2/etc. => protocolo del paper intacto.
        if algo_config.get('n_steps_laps_offset'):
            _off = int(algo_config['n_steps_laps_offset'])
            _per_env_laps = dynamic_n_steps // 2  # == n_laps (v1: n_steps = n_laps*2)
            dynamic_n_steps = (_per_env_laps + _off) * 2
            dynamic_batch_size = (_per_env_laps + _off) * n_envs * 2
            print(f"  n_steps_laps_offset={_off}: n_steps -> {dynamic_n_steps}, "
                  f"batch_size -> {dynamic_batch_size} (episodio = n_laps+1)")
        # V2: minibatch = one env's rollout (always divides the buffer);
        # V1: full-buffer batch (paper protocol)
        ppo_batch = dynamic_n_steps if algo_config.get('minibatch_per_env') else dynamic_batch_size
        # 'minibatches': split the rollout buffer into k minibatches. Without it
        # batch_size == n_steps*n_envs == the whole buffer, i.e. a single
        # full-batch update per epoch (paper protocol). sb3-contrib's recurrent
        # buffer accepts any batch size (it splits by sequence and pads), so this
        # is safe for the LSTM. See hyperparams/config_tuned.py.
        if algo_config.get('minibatches'):
            ppo_batch = max(1, dynamic_batch_size // int(algo_config['minibatches']))
        model_kwargs.update({
            'learning_rate': algo_config.get('learning_rate', 3e-4),
            'ent_coef': algo_config['ent_coef'],
            'batch_size': ppo_batch,
            'n_steps': dynamic_n_steps,
            'n_epochs': algo_config.get('n_epochs', 10),
            'clip_range': algo_config.get('clip_range', 0.2),
            'vf_coef': algo_config.get('vf_coef', 0.5),
            'max_grad_norm': algo_config.get('max_grad_norm', 0.5),
            'gae_lambda': algo_config.get('gae_lambda', 0.95),
        })
        if algo_config.get('target_kl') is not None:
            model_kwargs['target_kl'] = algo_config['target_kl']

    # Create and train model
    model = algo_cls(algo_config['policy'], vec_env, **model_kwargs)

    # Echo what was ACTUALLY used. With several hyperparameter sets and model
    # directories in this repo, a log that does not state its own configuration
    # cannot be attributed afterwards.
    _eff = {k: v for k, v in model_kwargs.items()
            if k not in ('tensorboard_log', 'policy_kwargs', 'device', 'verbose')}
    print(f"  Effective {algo_name} kwargs: {_eff}")
    print(f"  Effective policy_kwargs: {algo_config['policy_kwargs']}")
    if 'batch_size' in model_kwargs and 'n_steps' in model_kwargs:
        buf = model_kwargs['n_steps'] * n_envs
        print(f"  Rollout buffer {buf} = n_steps {model_kwargs['n_steps']} x n_envs {n_envs}"
              f"  ->  {buf / model_kwargs['batch_size']:.0f} minibatch(es) per epoch")

    # Entropy schedule (only when the selected hyperparameter set asks for one).
    callbacks = [eval_callback]
    if ent_target is not None:
        # P4: lazo cerrado sobre la entropía lograda. Excluye el anneal de lazo
        # abierto (pelearían por model.ent_coef).
        callbacks.append(EntropyTargetCallback(
            total_timesteps=total_timesteps, h_target=ent_target,
            k=ent_target_k, csv_path=os.path.join(algo_dir, 'entctl.csv'),
            verbose=1))
        print(f"  Entropy CONTROLLER (lazo cerrado): H_target={ent_target} "
              f"k={ent_target_k} (hold 70%, anneal a 0.03 en 90%)")
    elif algo_config.get('ent_coef_final') is not None:
        _frac = algo_config.get('ent_coef_anneal_frac', 1.0)
        callbacks.append(EntropyAnnealCallback(
            start=algo_config['ent_coef'], final=algo_config['ent_coef_final'],
            total_timesteps=total_timesteps, anneal_frac=_frac, verbose=1))
        print(f"  Entropy anneal: {algo_config['ent_coef']} -> "
              f"{algo_config['ent_coef_final']} over {_frac:.0%} of "
              f"{total_timesteps} steps, then held")
    if checkpoint_freq:
        # P3: checkpoints intermedios emparejados con sus stats de VecNormalize
        # (sb3 CheckpointCallback guarda ..._vecnormalize_<steps>_steps.pkl).
        ck_dir = os.path.join(algo_dir, 'ck')
        callbacks.append(CheckpointCallback(
            save_freq=checkpoint_freq, save_path=ck_dir, name_prefix='ck',
            save_vecnormalize=True, verbose=0))
        print(f"  Checkpoints: cada {checkpoint_freq} vec-steps -> {ck_dir}")

    model.learn(
        total_timesteps,
        callback=callbacks if len(callbacks) > 1 else eval_callback,
        progress_bar=True,
        tb_log_name=algo_name
    )

    # Save the FINAL VecNormalize stats for reference. The stats matching
    # best_model.zip are saved by SaveVecNormalizeOnBest as
    # vecnormalize_stats.pkl at the moment the best model is checkpointed.
    vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats_final.pkl')
    vec_env.save(vecnorm_path)
    # Safety net: if no evaluation improved (no best model saved yet),
    # fall back to the final stats.
    best_stats = os.path.join(algo_dir, 'vecnormalize_stats.pkl')
    if not os.path.exists(best_stats):
        vec_env.save(best_stats)

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

    # Hyperparameter set selection (v1 = paper protocol, v2 = revised)
    if args.hyperparams == 'v2':
        from hyperparams import config_v2 as hp2
        AVAILABLE_ALGORITHMS = {
            'DQN': (DoubleDQN, hp2.DQN_CONFIG),
            'A2C': (A2C, hp2.A2C_CONFIG),
            'TRPO': (TRPO, hp2.TRPO_CONFIG),
            'PPO': (PPO, hp2.PPO_CONFIG),
            'RecurrentPPO': (RecurrentPPO, hp2.RECURRENT_PPO_CONFIG),
        }
        TRAINING_CONFIG.update({
            'n_envs': hp2.TRAINING_CONFIG['n_envs'],
            'total_timesteps': hp2.TRAINING_CONFIG['total_timesteps'],
            'eval_freq': hp2.TRAINING_CONFIG['eval_freq'],
            'n_eval_episodes': hp2.TRAINING_CONFIG['n_eval_episodes'],
            'early_stop_patience': hp2.TRAINING_CONFIG['early_stop_patience'],
        })
        print("Using HYPERPARAMS V2 (see hyperparams/config_v2.py)")
    elif args.hyperparams != 'v1':
        # Any other name -> hyperparams/config_<name>.py, RecurrentPPO only, so
        # the other four algorithms keep the paper protocol and the comparison
        # isolates the RecurrentPPO change.
        import importlib
        hpt = importlib.import_module(f'hyperparams.config_{args.hyperparams}')
        AVAILABLE_ALGORITHMS = {
            'DQN': (DoubleDQN, DQN_CONFIG),
            'A2C': (A2C, A2C_CONFIG),
            'TRPO': (TRPO, TRPO_CONFIG),
            'PPO': (PPO, PPO_CONFIG),
            'RecurrentPPO': (RecurrentPPO, hpt.RECURRENT_PPO_CONFIG),
        }
        print(f"Using HYPERPARAMS '{args.hyperparams}' for RecurrentPPO "
              f"(hyperparams/config_{args.hyperparams}.py); other algorithms keep v1")

    # Output directory override (keeps the paper models intact for v2 runs)
    if args.output_dir:
        MODELS_DIR = args.output_dir
        os.makedirs(MODELS_DIR, exist_ok=True)

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
    n_envs = args.n_envs if args.n_envs is not None else TRAINING_CONFIG['n_envs']
    total_timesteps = args.timesteps if args.timesteps is not None else TRAINING_CONFIG['total_timesteps']
    eval_freq = args.eval_freq if args.eval_freq is not None else TRAINING_CONFIG['eval_freq']
    n_eval_episodes = TRAINING_CONFIG['n_eval_episodes']
    early_stop_patience = (args.early_stop_patience if args.early_stop_patience is not None
                           else TRAINING_CONFIG['early_stop_patience'])
    seed = args.seed

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    seed_all(seed)

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
    print(f"  Base seed: {seed}")

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
        env_factory = create_env_factory(gp, agent_driver, loader, initial_positions,
                                         fast_rivals=args.fast_env,
                                         pit_det_pass=args.pit_det_pass,
                                         following_gap_scale=args.following_gap_scale,
                                         sc_bunching=args.sc_bunching,
                                         sc_bunch_delta=args.sc_bunch_delta,
                                         rule_mask=args.rule_mask)
        if args.pit_det_pass or args.following_gap_scale or args.sc_bunching:
            print(f"  Env physics fixes: pit_det_pass={args.pit_det_pass} "
                  f"following_gap_scale={args.following_gap_scale} "
                  f"sc_bunching={args.sc_bunching} (delta={args.sc_bunch_delta})")

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
            checkpoint_freq=args.checkpoint_freq,
            norm_reward_off=args.norm_reward_off,
            ent_target=args.ent_target,
            ent_target_k=args.ent_target_k,
            norm_obs_off=args.norm_obs_off,
        )

        # Train selected algorithms for this GP (offset seed per algorithm so they don't share RNG state)
        for algo_idx, algo_name in enumerate(algorithms_to_train):
            algo_cls, algo_config = AVAILABLE_ALGORITHMS[algo_name]
            train_algorithm(algo_name, algo_cls, algo_config,
                            seed=seed + algo_idx,
                            **train_kwargs)

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
