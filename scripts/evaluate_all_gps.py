import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Evaluate trained RL models against real 2024 race data for ALL Grand Prix.

Generates a combined 2x5 grid boxplot comparing RL algorithm performance
across 10 GP/driver combinations.

Generalizes evaluate_bahrain_2024.py to all trained models.
"""

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
from dataclasses import dataclass

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO, TRPO

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader

# Must match the keys used during training (train_rl_agents.py)
NORM_OBS_KEYS = ['tyrelifes', 'positions', 'time_diff_to_agent']

N_SIMULATIONS = 500

# Algorithm display order, colors, and abbreviations
ALGO_ORDER = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']
ALGO_COLORS = {
    'DQN':          '#0072B2',  # blue       (Wong 2011)
    'A2C':          '#E69F00',  # orange
    'TRPO':         '#009E73',  # green
    'PPO':          '#CC79A7',  # pink
    'RecurrentPPO': '#56B4E9',  # sky blue
}
ALGO_ABBREV = {
    'DQN': 'DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'RPPO',
}
ALGO_CLASSES = {
    'RecurrentPPO': RecurrentPPO,
    'PPO': PPO,
    'DQN': DQN,
    'A2C': A2C,
    'TRPO': TRPO,
}


# =============================================================================
# GP CONFIG LOADING
# =============================================================================

def load_gp_config(gp_name: str) -> dict:
    """Load GP configuration from trained_models/simulation/{gp_name}/config.json.

    Returns dict with:
        initial_positions_list_2024, laps_vsc_2024, laps_sc_2024,
        initial_positions_2024
    """
    config_path = os.path.join(str(config.SIMULATION_DIR), gp_name, 'config.json')

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    return {
        'initial_positions_list_2024': cfg.get('initial_positions_list_2024', []),
        'laps_vsc_2024': cfg.get('laps_vsc_2024', []),
        'laps_sc_2024': cfg.get('laps_sc_2024', []),
        'initial_positions_2024': cfg.get('initial_positions_2024', {}),
    }


# =============================================================================
# GP CONFIGURATIONS
# =============================================================================
# Each entry defines one GP/driver evaluation.
#
# Compound legend per GP (from config.json "compounds" field):
#   Bahrain:        C1=1 (Hard), C2=2 (Medium), C3=3 (Soft)
#   Saudi Arabian:  C2=2 (Hard), C3=3 (Medium), C4=4 (Soft)
#   Miami:          C2=2 (Hard), C3=3 (Medium), C4=4 (Soft)
#   Emilia Romagna: C3=3 (Hard), C4=4 (Medium), C5=5 (Soft)
#   Hungarian:      C3=3 (Hard), C4=4 (Medium), C5=5 (Soft)
#   Belgian:        C2=2 (Hard), C3=3 (Medium), C4=4 (Soft)
#   Dutch:          C1=1 (Hard), C2=2 (Medium), C3=3 (Soft)
#   Singapore:      C3=3 (Hard), C4=4 (Medium), C5=5 (Soft)
#   United States:  C2=2 (Hard), C3=3 (Medium), C4=4 (Soft)
# =============================================================================

GP_CONFIGS = [
    # -------------------------------------------------------------------------
    # 1. Bahrain GP (ALO)
    # Compounds: C1=1 (Hard), C2=2 (Medium), C3=3 (Soft)
    # Key 0 = starting compound for each rival
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Bahrain Grand Prix',
        'agent_driver': 'Driver_ALO',
        'model_dir': 'Bahrain_ALO',
        'display_name': 'Bahrain GP (ALO)',
        'real_finishing_pos': 9,
        'rival_strategies': {
            'Driver_VER': {0: 3, 17: 1, 37: 3},
            'Driver_PER': {0: 3, 12: 1, 36: 3},
            'Driver_SAI': {0: 3, 14: 1, 35: 1},
            'Driver_LEC': {0: 3, 11: 1, 34: 1},
            'Driver_NOR': {0: 3, 13: 1, 33: 1},
            'Driver_HAM': {0: 3, 12: 1, 33: 1},
            'Driver_PIA': {0: 3, 12: 1, 34: 1},
            'Driver_RUS': {0: 3, 15: 1, 32: 1},
            'Driver_STR': {0: 3, 9: 1, 27: 1},
            'Driver_ZHO': {0: 3, 9: 1, 28: 1},
            'Driver_MAG': {0: 3, 11: 1, 32: 1},
            'Driver_RIC': {0: 3, 13: 1, 35: 3},
            'Driver_TSU': {0: 3, 14: 1, 34: 1},
            'Driver_ALB': {0: 3, 15: 1, 36: 1},
            'Driver_HUL': {0: 3, 1: 1, 20: 1, 41: 3},
            'Driver_OCO': {0: 3, 10: 1, 30: 1},
            'Driver_GAS': {0: 3, 12: 1, 31: 1, 43: 3},
            'Driver_BOT': {0: 3, 12: 1, 30: 1},
            'Driver_SAR': {0: 3, 10: 1, 28: 1, 40: 3},
        },
    },

    # -------------------------------------------------------------------------
    # 2. Saudi Arabian GP (NOR)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Saudi Arabian Grand Prix',
        'agent_driver': 'Driver_NOR',
        'model_dir': 'Saudi_Arabian_NOR',
        'display_name': 'Saudi Arabian GP (NOR)',
        'real_finishing_pos': 8,
        'rival_strategies': {
            'Driver_VER': {0: 3, 7: 2},
            'Driver_LEC': {0: 3, 7: 2},
            'Driver_PER': {0: 3, 7: 2},
            'Driver_ALO': {0: 3, 7: 2},
            'Driver_PIA': {0: 3, 7: 2},
            'Driver_RUS': {0: 3, 7: 2},
            'Driver_HAM': {0: 3, 36: 4},
            'Driver_TSU': {0: 3, 7: 2},
            'Driver_STR': {0: 3},  # DNF
            'Driver_SAI': {0: 4, 7: 2},
            'Driver_ALB': {0: 3, 7: 2},
            'Driver_MAG': {0: 3, 7: 2},
            'Driver_RIC': {0: 3, 7: 2},
            'Driver_HUL': {0: 3, 33: 2},
            'Driver_OCO': {0: 3, 7: 2},
            'Driver_GAS': {0: 3},  # DNF
            'Driver_BOT': {0: 4, 7: 2, 35: 4},
            'Driver_SAR': {0: 3, 7: 2},
            'Driver_ZHO': {0: 3, 41: 4},
        },
    },

    # -------------------------------------------------------------------------
    # 3. Miami GP (RUS)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Miami Grand Prix',
        'agent_driver': 'Driver_RUS',
        'model_dir': 'Miami_RUS',
        'display_name': 'Miami GP (RUS)',
        'real_finishing_pos': 8,
        'rival_strategies': {
            'Driver_VER': {0: 3, 23: 2},
            'Driver_LEC': {0: 3, 19: 2},
            'Driver_SAI': {0: 3, 27: 2},
            'Driver_PER': {0: 3, 17: 2, 28: 3},
            'Driver_NOR': {0: 3, 29: 2},
            'Driver_PIA': {0: 3, 27: 2, 40: 2},
            'Driver_HAM': {0: 3, 26: 2},
            'Driver_HUL': {0: 3, 12: 2, 28: 3},
            'Driver_TSU': {0: 3, 28: 2},
            'Driver_STR': {0: 3, 11: 2, 28: 3},
            'Driver_GAS': {0: 3, 12: 2},
            'Driver_OCO': {0: 3, 22: 2},
            'Driver_ALB': {0: 3, 10: 2, 53: 4},
            'Driver_ALO': {0: 2, 23: 3},
            'Driver_BOT': {0: 4, 11: 2, 29: 3},
            'Driver_SAR': {0: 3, 11: 2},       # DNF
            'Driver_MAG': {0: 2, 22: 3, 28: 3, 31: 3},
            'Driver_ZHO': {0: 3, 28: 4},
            'Driver_RIC': {0: 2, 28: 3},
        },
    },

    # -------------------------------------------------------------------------
    # 4. Emilia Romagna GP (TSU)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Emilia Romagna Grand Prix',
        'agent_driver': 'Driver_TSU',
        'model_dir': 'Emilia_Romagna_TSU',
        'display_name': 'Emilia Romagna GP (TSU)',
        'real_finishing_pos': 10,
        'rival_strategies': {
            'Driver_VER': {0: 4, 24: 3},
            'Driver_NOR': {0: 4, 22: 3},
            'Driver_LEC': {0: 4, 25: 3},
            'Driver_SAI': {0: 4, 27: 3},
            'Driver_PIA': {0: 4, 23: 3},
            'Driver_RUS': {0: 4, 21: 3, 52: 4},
            'Driver_HAM': {0: 4, 27: 3},
            'Driver_RIC': {0: 4, 11: 3},
            'Driver_HUL': {0: 4, 13: 3},
            'Driver_PER': {0: 3, 37: 4},
            'Driver_OCO': {0: 4, 25: 3},
            'Driver_STR': {0: 4, 37: 3},
            'Driver_ALB': {0: 4, 8: 3, 9: 4, 23: 4, 28: 4}, # DNF
            'Driver_GAS': {0: 5, 8: 3, 30: 4},
            'Driver_BOT': {0: 4, 8: 3},
            'Driver_ZHO': {0: 3, 33: 4},
            'Driver_MAG': {0: 4, 37: 3},
            'Driver_SAR': {0: 3, 31: 4},
            'Driver_ALO': {0: 5, 7: 3, 40: 4, 59: 5},
        },
    },

    # -------------------------------------------------------------------------
    # 5. Hungarian GP (SAI)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Hungarian Grand Prix',
        'agent_driver': 'Driver_SAI',
        'model_dir': 'Hungarian_SAI',
        'display_name': 'Hungarian GP (SAI)',
        'real_finishing_pos': 6,
        'rival_strategies': {
            'Driver_NOR': {0: 4, 17: 3, 45: 4},
            'Driver_PIA': {0: 4, 18: 3, 47: 4},
            'Driver_VER': {0: 4, 21: 3, 49: 4},
            'Driver_HAM': {0: 4, 16: 3, 40: 3},
            'Driver_LEC': {0: 4, 23: 3, 40: 4},
            'Driver_ALO': {0: 5, 7: 4, 37: 3},
            'Driver_STR': {0: 5, 14: 4, 45: 3},
            'Driver_RIC': {0: 4, 7: 3, 28: 3},
            'Driver_TSU': {0: 4, 29: 3},
            'Driver_HUL': {0: 4, 2: 3, 29: 3},
            'Driver_BOT': {0: 4, 16: 3, 45: 3},
            'Driver_ALB': {0: 5, 6: 3, 29: 3},
            'Driver_SAR': {0: 4, 8: 3, 33: 3, 63: 5},
            'Driver_MAG': {0: 5, 6: 3, 34: 3},
            'Driver_PER': {0: 3, 28: 4, 47: 4},
            'Driver_RUS': {0: 3, 33: 4, 53: 3},
            'Driver_ZHO': {0: 4, 7: 3, 36: 3},
            'Driver_OCO': {0: 4, 6: 3, 30: 3, 64: 5},
            'Driver_GAS': {0: 3, 28: 4},   # DNF
        },
    },

    # -------------------------------------------------------------------------
    # 6. Belgian GP (NOR)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Belgian Grand Prix',
        'agent_driver': 'Driver_NOR',
        'model_dir': 'Belgian_NOR',
        'display_name': 'Belgian GP (NOR)',
        'real_finishing_pos': 5,
        'rival_strategies': {
            'Driver_LEC': {0: 3, 12: 2, 25: 2},
            'Driver_PER': {0: 3, 11: 3, 21: 2, 42: 4},
            'Driver_HAM': {0: 3, 11: 2, 26: 2},
            'Driver_PIA': {0: 3, 11: 2, 30: 2},
            'Driver_RUS': {0: 3, 10: 2},  # DNF
            'Driver_SAI': {0: 2, 20: 3, 28: 2},
            'Driver_ALO': {0: 3, 13: 2},
            'Driver_OCO': {0: 3, 12: 2, 30: 2},
            'Driver_ALB': {0: 3, 8: 3, 23: 2},
            'Driver_VER': {0: 3, 10: 2, 28: 3},
            'Driver_GAS': {0: 3, 9: 2, 28: 2},
            'Driver_RIC': {0: 4, 8: 3, 21: 2},
            'Driver_BOT': {0: 3, 11: 2, 35: 3},
            'Driver_STR': {0: 3, 12: 2},
            'Driver_HUL': {0: 3, 7: 2, 20: 3},
            'Driver_MAG': {0: 3, 17: 2},
            'Driver_SAR': {0: 3, 8: 3, 24: 2},
            'Driver_ZHO': {0: 2},         # DNF
            'Driver_TSU': {0: 3, 15: 2},
        },
    },

    # -------------------------------------------------------------------------
    # 7. Dutch GP (RUS)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Dutch Grand Prix',
        'agent_driver': 'Driver_RUS',
        'model_dir': 'Dutch_RUS',
        'display_name': 'Dutch GP (RUS)',
        'real_finishing_pos': 7,
        'rival_strategies': {
            'Driver_NOR': {0: 2, 28: 1},
            'Driver_VER': {0: 2, 27: 1},
            'Driver_PIA': {0: 2, 33: 1},
            'Driver_PER': {0: 2, 29: 1},
            'Driver_LEC': {0: 2, 24: 1},
            'Driver_ALO': {0: 2, 31: 1},
            'Driver_STR': {0: 2, 30: 1},
            'Driver_GAS': {0: 2, 32: 1},
            'Driver_SAI': {0: 2, 30: 1},
            'Driver_TSU': {0: 3, 14: 2, 32: 1},
            'Driver_HUL': {0: 2, 14: 1},
            'Driver_RIC': {0: 2, 29: 1},
            'Driver_HAM': {0: 3, 23: 1, 48: 3},
            'Driver_OCO': {0: 2, 30: 1},
            'Driver_BOT': {0: 3, 15: 1, 43: 2},
            'Driver_ZHO': {0: 2, 18: 1, 51: 3},
            'Driver_SAR': {0: 2, 22: 1},
            'Driver_ALB': {0: 2, 12: 1, 54: 2},
            'Driver_MAG': {0: 1, 40: 2},
        },
    },

    # -------------------------------------------------------------------------
    # 8. Singapore GP (HUL)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'Singapore Grand Prix',
        'agent_driver': 'Driver_HUL',
        'model_dir': 'Singapore_HUL',
        'display_name': 'Singapore GP (HUL)',
        'real_finishing_pos': 9,
        'rival_strategies': {
            'Driver_NOR': {0: 4, 30: 3},
            'Driver_VER': {0: 4, 29: 3},
            'Driver_HAM': {0: 5, 17: 3},
            'Driver_RUS': {0: 4, 27: 3},
            'Driver_PIA': {0: 4, 38: 3},
            'Driver_ALO': {0: 4, 25: 3},
            'Driver_TSU': {0: 4, 33: 5},
            'Driver_LEC': {0: 4, 36: 3},
            'Driver_SAI': {0: 4, 13: 3},
            'Driver_ALB': {0: 4, 11: 3},  # DNF
            'Driver_SAR': {0: 4, 29: 3},
            'Driver_PER': {0: 4, 28: 3},
            'Driver_MAG': {0: 3, 28: 4, 49: 5},    # DNF
            'Driver_OCO': {0: 4, 29: 3},
            'Driver_RIC': {0: 5, 10: 4, 46: 5, 58: 5},
            'Driver_STR': {0: 3, 26: 4},
            'Driver_GAS': {0: 4, 37: 5},
            'Driver_BOT': {0: 3, 33: 4},
            'Driver_ZHO': {0: 3, 34: 4},
        },
    },

    # -------------------------------------------------------------------------
    # 9. United States GP (PIA)
    # -------------------------------------------------------------------------
    {
        'gp_name': 'United States Grand Prix',
        'agent_driver': 'Driver_PIA',
        'model_dir': 'United_States_PIA',
        'display_name': 'United States GP (PIA)',
        'real_finishing_pos': 5,
        'rival_strategies': {
            'Driver_NOR': {0: 3, 31: 2},
            'Driver_VER': {0: 3, 25: 2},
            'Driver_SAI': {0: 3, 21: 2},
            'Driver_LEC': {0: 3, 26: 2},
            'Driver_GAS': {0: 3, 18: 2},
            'Driver_ALO': {0: 3, 26: 2},
            'Driver_MAG': {0: 3, 17: 2, 38: 3},
            'Driver_PER': {0: 3, 26: 2},
            'Driver_TSU': {0: 3, 18: 2},
            'Driver_HUL': {0: 3, 27: 2},
            'Driver_OCO': {0: 3, 31: 2, 51: 4},
            'Driver_STR': {0: 2, 27: 3},
            'Driver_ALB': {0: 3, 3: 2, 33: 2},
            'Driver_SAR': {0: 2, 39: 3},
            'Driver_BOT': {0: 3, 15: 2},
            'Driver_HAM': {0: 2},     # DNF
            'Driver_ZHO': {0: 3, 13: 3, 35: 2},
            'Driver_RIC': {0: 2, 36: 3},
            'Driver_RUS': {0: 2, 40: 3},
        },
    },
]


# =============================================================================
# BUILD RIVAL STRATEGIES
# =============================================================================

def build_rival_strategies(config_entry: dict) -> tuple:
    """Strip agent driver, extract initial compounds (key 0), validate.

    Returns:
        (rival_actions, initial_compounds) where:
        - rival_actions: {driver: {pit_lap: compound}} for set_evaluation_mode
        - initial_compounds: {driver: compound_number} for race start

    Raises ValueError if any compound value is None (incomplete entry).
    """
    strategies = dict(config_entry['rival_strategies'])
    agent = config_entry['agent_driver']

    # Remove agent if accidentally included
    strategies.pop(agent, None)

    # Extract initial compounds (key 0) and build pit-stop-only strategies
    initial_compounds = {}
    rival_actions = {}
    for driver, stops in strategies.items():
        stops = dict(stops)  # copy
        if 0 in stops:
            initial_compounds[driver] = stops.pop(0)
        rival_actions[driver] = stops

    # Validate no compound is None
    for driver, compound in initial_compounds.items():
        if compound is None:
            raise ValueError(
                f"Incomplete rival strategy for {config_entry['model_dir']}: "
                f"{driver} initial compound (key 0) is None. "
                f"Fill in the compound number in GP_CONFIGS."
            )
    for driver, stops in rival_actions.items():
        for lap, compound in stops.items():
            if compound is None:
                raise ValueError(
                    f"Incomplete rival strategy for {config_entry['model_dir']}: "
                    f"{driver} pit lap {lap} has no compound. "
                    f"Fill in the compound number in GP_CONFIGS."
                )

    return rival_actions, initial_compounds


# =============================================================================
# SIMULATION
# =============================================================================

@dataclass
class SimResult:
    final_position: int
    total_reward: float
    pit_stops: int
    changed_compound: bool


def run_single_race(
    model,
    vec_env,
    raw_env,
    algo_name: str,
    seed: int,
    initial_compounds: Dict[str, int],
    deterministic_policy: bool = False,
) -> SimResult:
    """Run a single race evaluation with real rival strategies.

    Args:
        model: Trained SB3 model
        vec_env: VecNormalize-wrapped environment (for model.predict)
        raw_env: Underlying F1EnvAllDrivers (for attribute access)
        algo_name: Algorithm name
        seed: Random seed
        initial_compounds: Per-rival starting compound {driver_name: compound}
        deterministic_policy: Whether to use deterministic actions
    """
    # Seed all RNGs for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)
    obs = vec_env.reset()

    # Set each rival's starting compound individually
    for rival in raw_env.rival_drivers:
        compound = initial_compounds.get(rival.name)
        if compound is not None:
            rival.compound = compound
            rival.initial_compound = compound

    # LSTM states for RecurrentPPO
    use_lstm = (algo_name == 'RecurrentPPO')
    if use_lstm:
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
    else:
        lstm_states = None
        episode_start = None

    done = False
    total_reward = 0.0
    pit_stops = 0
    current_compound = None
    terminal_info = {}

    while not done:
        if use_lstm:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic_policy
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=deterministic_policy)

        # Track state BEFORE step (raw_env is wiped by DummyVecEnv auto-reset)
        is_lap_zero = (raw_env.lap_number == 0)
        agent_compound = raw_env.agent_driver.compound

        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        done = done[0]

        # Track pit stops.
        # Lap 0 is the starting compound selection (not a pit stop): initialize
        # current_compound from the compound chosen by the agent at lap 0.
        # Subsequent compound changes are real pit stops.
        if is_lap_zero:
            current_compound = raw_env.agent_driver.compound
        elif current_compound is None:
            current_compound = agent_compound
        elif agent_compound != current_compound:
            pit_stops += 1
            current_compound = agent_compound

        # Capture terminal info
        if done and 'final_position' in info[0]:
            terminal_info = info[0]

    changed_compound = terminal_info.get('has_changed_compound', False)
    raw_final_pos = terminal_info.get('final_position', raw_env.agent_position)
    # DSQ (no compound change) -> assign last place (P20)
    final_pos = 20 if not changed_compound else raw_final_pos

    return SimResult(
        final_position=final_pos,
        total_reward=total_reward,
        pit_stops=pit_stops,
        changed_compound=changed_compound,
    )


def evaluate_algorithm(
    algo_name: str,
    model_path: str,
    loader: ModelLoader,
    gp_name: str,
    agent_driver: str,
    initial_positions: List[int],
    rival_strategies: dict,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_compounds: Dict[str, int],
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 0,
) -> List[SimResult]:
    """Evaluate a single algorithm over N simulations for a given GP."""
    # Create raw environment
    raw_env = create_f1_env_all_drivers(
        gp=gp_name,
        driver=agent_driver,
        deterministic=False,
        yf_enabled=True,
        initial_positions=initial_positions,
        loader=loader,
        verbose=False,
    )

    # Set evaluation mode
    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions=rival_strategies,
    )

    # Wrap in DummyVecEnv + VecNormalize
    vec_env = DummyVecEnv([lambda: raw_env])

    # Load VecNormalize stats
    algo_dir = os.path.dirname(model_path)
    vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats.pkl')

    if os.path.exists(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"    Loaded VecNormalize stats from {vecnorm_path}")
        except (ValueError, AttributeError) as e:
            print(f"    WARNING: Failed to load VecNormalize stats ({e})")
            print(f"    Creating fresh VecNormalize with same config as training")
            vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
            vec_env.training = False
    else:
        print(f"    WARNING: No VecNormalize stats found at {vecnorm_path}")
        vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
        vec_env.training = False

    # Load model
    AlgoClass = ALGO_CLASSES[algo_name]
    model = AlgoClass.load(model_path, env=vec_env)

    # Verify model type
    model_class_name = type(model).__name__
    print(f"    Loaded model type: {model_class_name}")
    if algo_name == 'RecurrentPPO' and model_class_name != 'RecurrentPPO':
        print(f"    WARNING: Expected RecurrentPPO but got {model_class_name}")
    elif algo_name != 'RecurrentPPO' and model_class_name == 'RecurrentPPO':
        print(f"    WARNING: Expected {algo_name} but got RecurrentPPO")

    results = []
    for i in range(n_simulations):
        result = run_single_race(model, vec_env, raw_env, algo_name, seed=base_seed + i,
                                 initial_compounds=initial_compounds)
        results.append(result)

        if i == 0 and algo_name == 'RecurrentPPO':
            print(f"    [DEBUG] First simulation: P{result.final_position}, "
                  f"Pits: {result.pit_stops}, Changed: {result.changed_compound}")

        if (i + 1) % 25 == 0:
            positions = [r.final_position for r in results]
            print(f"    {i+1}/{n_simulations}  |  "
                  f"Mean pos: {np.mean(positions):.2f}  |  "
                  f"Best: P{min(positions)}  |  "
                  f"DSQ: {sum(1 for r in results if not r.changed_compound)}")

    vec_env.close()
    return results


# =============================================================================
# EVALUATE A SINGLE GP
# =============================================================================

def evaluate_gp(
    config_entry: dict,
    loader: ModelLoader,
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 0,
) -> Dict[str, List[SimResult]]:
    """Evaluate all 5 algorithms for a single GP/driver combination.

    Returns dict mapping algorithm name to list of SimResult.
    """
    gp_config = load_gp_config(config_entry['gp_name'])
    rival_strategies, initial_compounds = build_rival_strategies(config_entry)

    initial_positions = gp_config['initial_positions_list_2024']
    laps_vsc = gp_config['laps_vsc_2024']
    laps_sc = gp_config['laps_sc_2024']

    # Derive starting position from config
    positions_2024 = gp_config['initial_positions_2024']
    agent = config_entry['agent_driver']
    starting_pos = positions_2024.get(agent, '?')
    finishing_pos = config_entry['real_finishing_pos']

    model_dir = config_entry['model_dir']
    models_base = os.path.join(str(config.RL_AGENTS_DIR), model_dir)

    print(f"\n{'=' * 70}")
    print(f"GP: {config_entry['display_name']}")
    print(f"{'=' * 70}")
    print(f"  Agent: {agent}  |  Start: P{starting_pos}  |  Real finish: P{finishing_pos}")
    print(f"  SC laps: {laps_sc}  |  VSC laps: {laps_vsc}")
    print(f"  Simulations per algorithm: {n_simulations}")

    all_results: Dict[str, List[SimResult]] = {}

    for algo in ALGO_ORDER:
        model_path = os.path.join(models_base, algo, 'best_model.zip')

        if not os.path.exists(model_path):
            print(f"\n  [{algo}] Model not found at {model_path}, skipping.")
            continue

        print(f"\n{'─' * 70}")
        print(f"  Evaluating {algo}...")
        print(f"{'─' * 70}")

        results = evaluate_algorithm(
            algo_name=algo,
            model_path=model_path,
            loader=loader,
            gp_name=config_entry['gp_name'],
            agent_driver=agent,
            initial_positions=initial_positions,
            rival_strategies=rival_strategies,
            laps_vsc=laps_vsc,
            laps_sc=laps_sc,
            initial_compounds=initial_compounds,
            n_simulations=n_simulations,
            base_seed=base_seed,
        )
        all_results[algo] = results

        # Per-algorithm summary
        positions = [r.final_position for r in results]
        dsq_count = sum(1 for r in results if not r.changed_compound)
        mean_pits = np.mean([r.pit_stops for r in results])

        print(f"\n  {algo} Summary:")
        print(f"    Mean position:  {np.mean(positions):.2f} +/- {np.std(positions):.2f}")
        print(f"    Median:         P{int(np.median(positions))}")
        print(f"    Best / Worst:   P{min(positions)} / P{max(positions)}")
        print(f"    Mean pit stops: {mean_pits:.2f}")
        print(f"    DSQ rate:       {dsq_count}/{len(results)}")
        if finishing_pos is not None:
            beat_real = sum(1 for p in positions if p < finishing_pos)
            print(f"    Beat real P{finishing_pos}:   {beat_real}/{len(results)} "
                  f"({beat_real/len(results)*100:.0f}%)")

    return all_results


# =============================================================================
# GRID BOXPLOT
# =============================================================================

def plot_grid_boxplot(
    all_gp_results: List[dict],
    save_path: Optional[str] = None,
):
    """Generate 3x3 grid boxplot of all GP evaluations.

    Args:
        all_gp_results: List of dicts with keys:
            'display_name', 'real_starting_pos', 'real_finishing_pos',
            'results' (Dict[algo, List[SimResult]])
        save_path: Path to save the figure
    """
    n_gps = len(all_gp_results)
    nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18), sharey=True)
    axes_flat = axes.flatten()

    for idx, gp_data in enumerate(all_gp_results):
        ax = axes_flat[idx]
        results = gp_data['results']

        # Filter to algorithms that have results, in order
        available = [a for a in ALGO_ORDER if a in results]
        position_data = [[r.final_position for r in results[a]] for a in available]
        tick_labels = [ALGO_ABBREV.get(a, a) for a in available]

        if position_data:
            bp = ax.boxplot(
                position_data,
                labels=tick_labels,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(
                    marker='D', markerfacecolor='red',
                    markeredgecolor='red', markersize=6,
                ),
                showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='none',
                                markeredgecolor='gray', markersize=4,
                                linestyle='none'),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1),
            )

            # Color boxes
            for patch, algo in zip(bp['boxes'], available):
                color = ALGO_COLORS.get(algo, '#CCCCCC')
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

        # Reference lines
        starting_pos = gp_data.get('real_starting_pos')
        finishing_pos = gp_data.get('real_finishing_pos')

        if starting_pos is not None:
            offset = -0.15 if (finishing_pos is not None and starting_pos == finishing_pos) else 0
            ax.axhline(y=starting_pos + offset, color='red', linestyle='--',
                       linewidth=1.5, zorder=0)
        if finishing_pos is not None:
            ax.axhline(y=finishing_pos, color='blue', linestyle='--',
                       linewidth=1.5, zorder=0)

        # Formatting
        ax.set_title(gp_data['display_name'], fontsize=14, fontweight='bold')
        ax.set_ylim(0.5, 20.5)
        ax.set_yticks(range(1, 21))
        ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.3, linestyle='-')
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Y-axis label only on leftmost column
        if idx % ncols == 0:
            ax.set_ylabel('Agent Final Position', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_gps, nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label='Starting Position'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5,
               label='Real Final Position'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markeredgecolor='red', markersize=8, label='Mean'),
    ]

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_elements),
        fontsize=12,
        frameon=True,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nGrid plot saved to: {save_path}")


# =============================================================================
# LATEX TABLE
# =============================================================================

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

ALGO_DISPLAY = {
    'DQN': 'Double DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'Recurrent PPO',
}


def generate_latex_table(all_gp_results: List[dict]) -> str:
    """Aggregate positions and points across all GPs and return a LaTeX table."""
    # Collect all positions per algorithm across all GPs
    all_positions: Dict[str, List[int]] = {a: [] for a in ALGO_ORDER}

    for gp_data in all_gp_results:
        for algo, sim_results in gp_data['results'].items():
            for r in sim_results:
                all_positions[algo].append(r.final_position)

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{ccccc}')
    lines.append(r'\toprule')
    lines.append(r'\multicolumn{1}{l}{}               & \multicolumn{2}{c}{Position}   & \multicolumn{2}{c}{Points} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    lines.append(r'\multicolumn{1}{c}{Model}          &    $\mu$ & $\sigma$ &   $\mu$  & $\sigma$ \\ \midrule')

    for algo in ALGO_ORDER:
        positions = np.array(all_positions[algo])
        if len(positions) == 0:
            continue
        points = np.array([F1_POINTS.get(int(p), 0) for p in positions])

        mu_pos  = np.mean(positions)
        std_pos = np.std(positions)
        mu_pts  = np.mean(points)
        std_pts = np.std(points)

        display = ALGO_DISPLAY[algo]
        lines.append(
            rf'\multicolumn{{1}}{{c}}{{{display:<14}}} & '
            rf'{mu_pos:>8.2f}  & {std_pos:>8.2f}  & '
            rf'{mu_pts:>8.2f}  & {std_pts:>8.2f}  \\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Average positions and points obtained by each model, along with their standard deviations on selected circuits after 500 simulations per model for the 2024 Season.}')
    lines.append(r'\label{tab:tab_puntos_pos_mean_std_2024}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def is_entry_complete(entry: dict) -> bool:
    """Check if a GP_CONFIGS entry has all required fields filled in."""
    if entry['real_finishing_pos'] is None:
        return False
    # Check rival strategies for None compounds (including key 0)
    for driver, stops in entry['rival_strategies'].items():
        for lap, compound in stops.items():
            if compound is None:
                return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate RL agents against real 2024 race data for all GPs'
    )
    parser.add_argument(
        '--n-sims', type=int, default=N_SIMULATIONS,
        help=f'Number of simulations per algorithm (default: {N_SIMULATIONS})'
    )
    parser.add_argument(
        '--gps', nargs='+', default=None,
        help='Filter by model_dir name (e.g., Bahrain_ALO Belgian_NOR)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Base random seed for reproducibility (default: 0)'
    )
    args = parser.parse_args()

    # Filter GP configs
    configs = GP_CONFIGS
    if args.gps:
        configs = [c for c in configs if c['model_dir'] in args.gps]
        if not configs:
            print(f"No matching GP configs for: {args.gps}")
            print(f"Available: {[c['model_dir'] for c in GP_CONFIGS]}")
            sys.exit(1)

    print("=" * 70)
    print("EVALUATION: RL Agents vs Real 2024 Race Results")
    print("=" * 70)
    print(f"  GPs to evaluate: {len(configs)}")
    print(f"  Simulations per algorithm: {args.n_sims}")
    print(f"  Base seed: {args.seed}")
    print("=" * 70)

    loader = ModelLoader()
    all_gp_results = []

    for entry in configs:
        # Check completeness
        if not is_entry_complete(entry):
            print(f"\n  WARNING: Skipping {entry['model_dir']} -- "
                  f"incomplete data (None values in rival_strategies "
                  f"or real_finishing_pos).")
            continue

        # Run evaluation
        results = evaluate_gp(entry, loader, n_simulations=args.n_sims, base_seed=args.seed)

        if not results:
            print(f"\n  WARNING: No models found for {entry['model_dir']}, skipping.")
            continue

        # Load config for starting position
        gp_config = load_gp_config(entry['gp_name'])
        starting_pos = gp_config['initial_positions_2024'].get(entry['agent_driver'])

        all_gp_results.append({
            'display_name': entry['display_name'],
            'real_starting_pos': starting_pos,
            'real_finishing_pos': entry['real_finishing_pos'],
            'results': results,
        })

        # Print comparison table for this GP
        print(f"\n{'=' * 70}")
        print(f"COMPARISON TABLE -- {entry['display_name']}")
        print(f"{'=' * 70}")
        finishing = entry['real_finishing_pos']
        print(f"{'Algorithm':<15} {'Mean Pos':<12} {'Median':<8} {'Best':<6} "
              f"{'Pits':<6} {'Beat Real':<10}")
        print("-" * 70)

        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            res = results[algo]
            positions = [r.final_position for r in res]
            mean_pits = np.mean([r.pit_stops for r in res])
            beat = sum(1 for p in positions if p < finishing)
            print(f"{algo:<15} "
                  f"{np.mean(positions):>5.2f} +/- {np.std(positions):<4.1f} "
                  f"P{int(np.median(positions)):<5} "
                  f"P{min(positions):<4} "
                  f"{mean_pits:<5.1f} "
                  f"{beat:>3}/{len(res)} ({beat/len(res)*100:>4.0f}%)")
        print("-" * 70)

    if not all_gp_results:
        print("\nNo complete GP evaluations. Fill in TODO entries in GP_CONFIGS.")
        sys.exit(1)

    # Generate grid plot
    output_path = os.path.join(str(config.RL_AGENTS_DIR), 'all_gps_2024_boxplot.pdf')
    plot_grid_boxplot(all_gp_results, save_path=output_path)

    # Generate LaTeX table
    latex_table = generate_latex_table(all_gp_results)
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex_table)

    table_path = os.path.join(str(config.RL_AGENTS_DIR), 'all_gps_2024_table.tex')
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {table_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"  GPs evaluated: {len(all_gp_results)}")
    print(f"  GPs skipped: {len(configs) - len(all_gp_results)}")
    print("=" * 70)
