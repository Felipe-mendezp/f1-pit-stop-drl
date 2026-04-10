"""
F1 Multi-Driver Environment (All Drivers)

Complete implementation of a Formula 1 race simulation environment with:
- 20 drivers (1 agent + 19 rivals)
- FastLinearPredictor for 32x faster lap time predictions
- Overtake simulation (Algorithm 2 from paper)
- Rival strategies: pit stop logit + compound choice conditional logit
- SC/VSC events with configurable probabilities

Follows mandatory state transition equations exactly.
Compatible with Stable Baselines 3 and Gymnasium.
"""

import sys
import os
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Dict as GymDict, Discrete, Box, MultiBinary

import config  # setup sys.path

from utils import (
    initial_lap_v3,
    update_lap_state_v3,
    build_yf_features,
)
from reg import FastLinearPredictor
from model_loader import ModelLoader, GPModels


# === CONSTANTS ===

LAPS_DD = {
    'Bahrain Grand Prix': 57,
    'Belgian Grand Prix': 44,
    'Dutch Grand Prix': 72,
    'Emilia Romagna Grand Prix': 63,
    'Hungarian Grand Prix': 70,
    'Miami Grand Prix': 57,
    'Saudi Arabian Grand Prix': 50,
    'Singapore Grand Prix': 62,
    'United States Grand Prix': 56,
}

GP_COMPOUNDS = {
    'Bahrain Grand Prix': ['C1', 'C2', 'C3'],
    'Belgian Grand Prix': ['C2', 'C3', 'C4'],
    'Dutch Grand Prix': ['C1', 'C2', 'C3'],
    'Emilia Romagna Grand Prix': ['C3', 'C4', 'C5'],
    'Hungarian Grand Prix': ['C3', 'C4', 'C5'],
    'Miami Grand Prix': ['C2', 'C3', 'C4'],
    'Saudi Arabian Grand Prix': ['C2', 'C3', 'C4'],
    'Singapore Grand Prix': ['C3', 'C4', 'C5'],
    'United States Grand Prix': ['C2', 'C3', 'C4'],
}

MIN_GAP = 0.4  # Minimum gap between cars (seconds)
VSC_DURATION = 2  # VSC lasts 2 laps
SC_DURATION = 4   # SC lasts 4 laps
MIN_STINT_LENGTH = 8  # Minimum stint length for rivals (unless SC/VSC)
POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ALL_DRIVERS = {
    'Driver_ALB': 23, 'Driver_ALO': 14, 'Driver_BOT': 77,
    'Driver_GAS': 10, 'Driver_HAM': 44, 'Driver_HUL': 27,
    'Driver_LEC': 16, 'Driver_MAG': 20, 'Driver_NOR': 4,
    'Driver_OCO': 31, 'Driver_PER': 11, 'Driver_PIA': 81,
    'Driver_RIC': 3, 'Driver_RUS': 63, 'Driver_SAI': 55,
    'Driver_SAR': 2, 'Driver_STR': 18, 'Driver_TSU': 22,
    'Driver_VER': 1, 'Driver_ZHO': 24
}


@dataclass
class Driver:
    """Encapsulates all state for a single driver."""
    name: str
    initial_pos: int
    position: int
    compound: int
    initial_compound: int
    tyrelife: int = 0
    total_time: float = 0.0
    has_changed_compound: bool = False
    is_dnf: bool = False
    lap_state: Dict[str, Any] = field(default_factory=dict)
    prev_action: int = 0
    predicted_laptime: float = 0.0


class F1EnvAllDrivers(Env):
    """
    Formula 1 race simulation environment with 20 drivers.

    Features:
    - FastLinearPredictor for 32x faster lap times (Equation 1)
    - Overtake simulation using logit model (Algorithm 2, Equation 3)
    - Yellow flag lap times (Equation 4)
    - Rival pit stop decisions using logit (Equation 5)
    - Compound choice using conditional logit
    - Observation cache for performance
    - __slots__ optimization
    """

    __slots__ = (
        # Configuration
        'n_laps', 'available_compounds', 'min_compound', 'max_compound',
        'gp', 'agent_driver_name', '_n_compounds', 'gp_compounds',
        # Models
        'fast_predictor_clear', 'fast_predictor_yf',
        'logit_overtake', 'rival_pitstop_logit', 'rival_compound_logit',
        # Probabilities and thresholds
        'race_vsc_prob', 'race_sc_prob', 'lap_vsc_prob', 'lap_sc_prob',
        'overtake_threshold', 'lap_time_noise_std',
        # Options
        'deterministic', 'yf_enabled', 'initial_positions', 'verbose',
        # Drivers
        'drivers', 'agent_driver', 'rival_drivers',
        # Global state
        'lap_number', 'laps_left', 'vsc', 'sc',
        'laps_left_vsc', 'laps_left_sc',
        # Reward and penalty constants
        '_penalty_disqualification', '_penalty_pit_last_lap',
        '_penalty_early_pit', '_penalty_consecutive_pit', '_penalty_invalid',
        '_early_laps_set', '_points_multiplier', '_position_multiplier',
        # Cached column names and O(1) lookup maps
        '_compound_cols', '_tyre_life_cols',
        '_compound_col_map', '_tyre_life_col_map',
        # Observation cache
        '_obs_cache', 'observation_space', 'action_space',
        # Reward type
        'reward_type',
        # Evaluation mode
        'eval', 'laps_vsc', 'laps_sc', 'rival_actions_dict',
        # Last laptime for reward
        '_agent_last_laptime',
        # Baseline laptime for centered 'time' reward
        '_baseline_laptime',
    )

    def __init__(
        self,
        n_laps: int,
        available_compounds: List[int],
        gp: str,
        agent_driver: str,
        fast_predictor_clear: FastLinearPredictor,
        fast_predictor_yf=None,
        logit_overtake=None,
        rival_pitstop_logit=None,
        rival_compound_logit=None,
        vsc_prob: float = 0.0,
        sc_prob: float = 0.0,
        overtake_threshold: float = 0.5,
        deterministic: bool = False,
        yf_enabled: bool = True,
        initial_positions: Optional[List[int]] = None,
        lap_time_noise_std: float = 0.3,
        verbose: bool = False,
        reward_type: str = 'mix',
    ):
        """
        Initialize F1 multi-driver environment.

        Args:
            n_laps: Total race laps
            available_compounds: List of compound numbers (e.g., [2, 3, 4])
            gp: Grand Prix name
            agent_driver: Agent's driver name (e.g., 'Driver_VER')
            fast_predictor_clear: FastLinearPredictor for clear track lap times
            fast_predictor_yf: FastLinearPredictor/statsmodels for yellow flag lap times
            logit_overtake: Overtake prediction model
            rival_pitstop_logit: Logit model for rival pit stop decisions
            rival_compound_logit: Conditional logit for compound choice
            vsc_prob: Probability of VSC per race
            sc_prob: Probability of SC per race
            overtake_threshold: Threshold for overtake logit
            deterministic: If True, no randomness in lap times
            yf_enabled: If True, simulate SC/VSC events
            initial_positions: Fixed starting positions (for evaluation)
            lap_time_noise_std: Std dev for lap time noise
            verbose: Print debug information
            reward_type: Reward structure ('mix', 'time', 'position', 'points')
        """
        super().__init__()

        # Configuration
        self.n_laps = n_laps
        self.available_compounds = [
            c if isinstance(c, int) else int(c[1:])
            for c in available_compounds
        ]
        self._n_compounds = len(self.available_compounds)
        self.min_compound = min(self.available_compounds)
        self.max_compound = max(self.available_compounds)
        self.gp = gp
        self.gp_compounds = [f'C{c}' for c in self.available_compounds]
        self.agent_driver_name = agent_driver

        # Models
        self.fast_predictor_clear = fast_predictor_clear
        self.fast_predictor_yf = fast_predictor_yf
        self.logit_overtake = logit_overtake
        self.rival_pitstop_logit = rival_pitstop_logit
        self.rival_compound_logit = rival_compound_logit

        # Probabilities
        self.race_vsc_prob = vsc_prob
        self.race_sc_prob = sc_prob
        self.lap_vsc_prob = 1 - (1 - vsc_prob) ** (1 / n_laps) if vsc_prob > 0 else 0.0
        self.lap_sc_prob = 1 - (1 - sc_prob) ** (1 / n_laps) if sc_prob > 0 else 0.0

        # Thresholds and options
        self.overtake_threshold = overtake_threshold
        self.deterministic = deterministic
        self.yf_enabled = yf_enabled
        self.initial_positions = initial_positions
        self.lap_time_noise_std = lap_time_noise_std
        self.verbose = verbose

        # Reward type validation
        valid_reward_types = {'mix', 'time', 'position', 'points'}
        if reward_type not in valid_reward_types:
            raise ValueError(f"reward_type must be one of {valid_reward_types}, got '{reward_type}'")
        self.reward_type = reward_type

        # Reward and penalty constants (scaled per reward type)
        self._early_laps_set = frozenset([1, 2, 3, 4, 5])

        if reward_type == 'mix':
            # Lap-time scale: ~-90/step, total ~-5130, plus large terminal bonuses
            self._penalty_disqualification = -5000.0
            self._penalty_pit_last_lap = -150.0
            self._penalty_early_pit = -100.0
            self._penalty_consecutive_pit = -200.0
            self._penalty_invalid = -600.0
        elif reward_type == 'time':
            # Centered per-lap scale: per-step ~[-5, +5], episode ~[-300, +300]
            self._penalty_disqualification = -500.0
            self._penalty_pit_last_lap = -30.0
            self._penalty_early_pit = -20.0
            self._penalty_consecutive_pit = -50.0
            self._penalty_invalid = -100.0
        elif reward_type == 'position':
            # Position scale: terminal in [-5, -100] (x5 multiplier)
            # DQ must be strictly worse than P20 (-100), use -200
            self._penalty_disqualification = -200.0
            self._penalty_pit_last_lap = -10.0
            self._penalty_early_pit = -5.0
            self._penalty_consecutive_pit = -15.0
            self._penalty_invalid = -25.0
        else:  # 'points'
            # Points scale: terminal in [0, 25]
            self._penalty_disqualification = -50.0
            self._penalty_pit_last_lap = -3.0
            self._penalty_early_pit = -2.0
            self._penalty_consecutive_pit = -5.0
            self._penalty_invalid = -10.0

        # Points multiplier for top-10 (only used by 'mix')
        self._points_multiplier = 300.0

        # Position multiplier for ALL positions (only used by 'mix')
        self._position_multiplier = 150.0

        # Drivers (initialized in reset)
        self.drivers: List[Driver] = []
        self.agent_driver: Optional[Driver] = None
        self.rival_drivers: List[Driver] = []

        # Global state
        self.lap_number = 0
        self.laps_left = n_laps
        self.vsc = 0
        self.sc = 0
        self.laps_left_vsc = VSC_DURATION
        self.laps_left_sc = SC_DURATION

        # Cached column names and O(1) lookup maps (set during initialization)
        self._compound_cols = None
        self._tyre_life_cols = None
        self._compound_col_map = {}
        self._tyre_life_col_map = {}

        # Evaluation mode
        self.eval = False
        self.laps_vsc = []
        self.laps_sc = []
        self.rival_actions_dict = {}

        # Last laptime for reward calculation
        self._agent_last_laptime = 0.0
        self._baseline_laptime = 0.0

        # Observation cache
        self._obs_cache = self._create_obs_cache()

        # Spaces
        self.observation_space = GymDict({
            "compounds": Box(low=0, high=self.max_compound, shape=(20,), dtype=np.float32),
            "tyrelifes": Box(low=0, high=n_laps + 1, shape=(20,), dtype=np.float32),
            "change_compounds": MultiBinary(20),
            "positions": Box(low=1, high=20, shape=(20,), dtype=np.float32),
            "time_diff_to_agent": Box(low=-500, high=500, shape=(19,), dtype=np.float32),
            "lap_number": Discrete(n_laps + 2),
            "laps_left": Discrete(n_laps + 2),
            "prev_action": Discrete(self.max_compound + 1),
            "vsc": Discrete(3),
            "sc": Discrete(5),
        })

        # Action space: 0=no pit, 1-3=compounds (hardest to softest)
        self.action_space = Discrete(4)

        if verbose:
            self._log_init_info()

    def _create_obs_cache(self) -> Dict[str, Any]:
        """Create initial observation cache."""
        return {
            "compounds": np.zeros(20, dtype=np.float32),
            "tyrelifes": np.zeros(20, dtype=np.float32),
            "change_compounds": np.zeros(20, dtype=np.int64),
            "positions": np.zeros(20, dtype=np.float32),
            "time_diff_to_agent": np.zeros(19, dtype=np.float32),
            "lap_number": 0,
            "laps_left": self.n_laps,
            "prev_action": 0,
            "vsc": 0,
            "sc": 0,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._initialize_race_state()
        self._update_obs_cache()
        return self._obs_cache.copy(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Per-Lap Evolution Order (strict):
        1. _update_sc_vsc() - Sample/update SC/VSC status
        2. _simulate_rival_pits(is_yf) - Rival pit decisions via logit
        3. _update_all_lap_states() - Update lap_state dicts
        4. _simulate_lap_times(is_yf) - Predict preliminary ŷ_d
        5. _resolve_overtakes() - Adjust ŷ_d and apply to cumulative time
        6. _update_positions_and_intervals() - Rank by cumulative time
        7. _update_driver_states() - Apply state transitions
        8. _update_safety_car_duration() - Decrement SC/VSC counters
        """
        # Handle lap 0 (starting compound selection) - special handling
        if self.lap_number == 0:
            # Pass action directly to _handle_lap_0, which will handle invalid actions
            return self._handle_lap_0(action)

        # Map action to compound (for laps > 0)
        if action == 0:
            agent_action = 0  # No pit
        elif 1 <= action <= self._n_compounds:
            agent_action = self.available_compounds[action - 1]
        else:
            # Invalid action during race
            self._update_obs_cache()
            return self._obs_cache.copy(), self._penalty_invalid, False, False, {}

        # === STRICT PER-LAP EVOLUTION ORDER ===

        # 1. Update safety car status
        is_yf = self._update_sc_vsc()

        # 2. Simulate rival pit decisions
        rival_actions = self._simulate_rival_pits(is_yf)

        # 3. Update all lap states
        self._update_all_lap_states(agent_action, rival_actions, is_yf)

        # 4. Simulate lap times (preliminary ŷ_d for all drivers)
        self._simulate_lap_times(is_yf)

        # 5. Resolve overtakes and apply lap times to cumulative total
        self._resolve_overtakes(is_yf)

        # 6. Update positions and intervals
        self._update_positions_and_intervals()

        # 7. Update driver states (apply mandatory state transitions)
        self._update_driver_states(agent_action, rival_actions, is_yf)

        # 8. Update safety car duration
        self._update_safety_car_duration()

        # Update global state
        self.lap_number += 1
        self.laps_left = max(0, self.n_laps - self.lap_number)

        # Check if race is complete
        done = self.lap_number >= self.n_laps

        # Calculate reward
        reward = self._calculate_reward(action, done)

        # Update observation cache
        self._update_obs_cache()

        # Include terminal info so evaluation can read it after DummyVecEnv auto-reset
        info = {}
        if done:
            info['final_position'] = self.agent_driver.position
            info['has_changed_compound'] = self.agent_driver.has_changed_compound
            info['total_time'] = self.agent_driver.total_time

        return self._obs_cache.copy(), reward, done, False, info

    def _handle_lap_0(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handle lap 0 - agent chooses starting compound."""
        # Map action to compound: action 0 is invalid, 1-3 map to available compounds
        if action == 0:
            # Invalid: must select a compound at lap 0
            # Assign random compound but still penalize
            penalty = self._penalty_invalid
            compound = self.np_random.choice(self.available_compounds)
        elif 1 <= action <= self._n_compounds:
            # Valid action: map to compound (softest to hardest)
            penalty = 0.0
            compound = self.available_compounds[action - 1]
        else:
            # Invalid action (outside valid range)
            # Assign random compound and penalize
            penalty = self._penalty_invalid
            compound = self.np_random.choice(self.available_compounds)

        # Set agent compound
        self.agent_driver.compound = compound
        self.agent_driver.initial_compound = compound

        # Initialize all lap states
        self._initialize_all_lap_states()

        # Progress to lap 1
        self.lap_number = 1
        self.laps_left = self.n_laps - 1

        # prev_action=0 means "no pit stop happened before this lap"
        # Setting it to the compound would incorrectly trigger PitOut logic on lap 1
        self.agent_driver.prev_action = 0

        # Update observation cache
        self._update_obs_cache()

        return self._obs_cache.copy(), penalty, False, False, {}

    def _initialize_race_state(self):
        """Initialize race state for all drivers."""
        self.lap_number = 0
        self.laps_left = self.n_laps
        self.laps_left_vsc = VSC_DURATION
        self.laps_left_sc = SC_DURATION
        self.vsc = 0
        self.sc = 0
        self._agent_last_laptime = 0.0

        driver_names = list(ALL_DRIVERS.keys())
        self.drivers = []
        self.rival_drivers = []

        # Set initial positions
        if self.initial_positions is not None and len(self.initial_positions) == 20:
            initial_positions = self.initial_positions[:]
        else:
            initial_positions = list(range(1, 21))
            self.np_random.shuffle(initial_positions)

        # Create agent driver
        agent_name = self.agent_driver_name
        agent_idx = driver_names.index(agent_name)
        agent_pos = initial_positions[agent_idx]
        self.agent_driver = Driver(
            name=agent_name,
            initial_pos=agent_pos,
            position=agent_pos,
            compound=0,
            initial_compound=0
        )
        self.drivers.append(self.agent_driver)

        # Create rival drivers
        for i, name in enumerate(driver_names):
            if name == agent_name:
                continue

            pos = initial_positions[driver_names.index(name)]
            compound = self.np_random.choice(self.available_compounds)

            driver = Driver(
                name=name,
                initial_pos=pos,
                position=pos,
                compound=compound,
                initial_compound=compound
            )
            self.drivers.append(driver)
            self.rival_drivers.append(driver)

        # Sort by position
        self.drivers.sort(key=lambda d: d.position)

    def _initialize_all_lap_states(self):
        """Initialize lap states for all drivers."""
        # Agent
        self.agent_driver.lap_state = initial_lap_v3(
            self.agent_driver.compound,
            self.agent_driver.name,
            self.fast_predictor_clear
        )
        self._initialize_first_lap_variables(self.agent_driver)

        # Compute baseline laptime for centered 'time' reward
        if self.reward_type == 'time':
            self._baseline_laptime = self.fast_predictor_clear.predict_single(
                self.agent_driver.lap_state
            )

        # Rivals
        for driver in self.rival_drivers:
            driver.lap_state = initial_lap_v3(
                driver.compound,
                driver.name,
                self.fast_predictor_clear
            )
            self._initialize_first_lap_variables(driver)

        # Cache column names for optimization
        if self.agent_driver.lap_state:
            self._compound_cols = sorted([
                col for col in self.agent_driver.lap_state.keys()
                if 'Compound_Detail' in col and 'TyreLife' not in col and '[' in col
            ])
            self._tyre_life_cols = sorted([
                col for col in self.agent_driver.lap_state.keys()
                if 'TyreLife' in col and ':' in col
            ])

            # Build O(1) lookup maps: compound_number -> column_name
            # This avoids linear search through column lists in update_lap_state_v3
            self._compound_col_map = {}  # {1: 'C(Compound_Detail)[T.C1]', 2: ...}
            self._tyre_life_col_map = {}  # {1: 'TyreLife:C(Compound_Detail)[T.C1]', ...}

            for col in self._compound_cols:
                for i in range(1, 6):
                    if f'C{i}' in col:
                        self._compound_col_map[i] = col
                        break

            for col in self._tyre_life_cols:
                for i in range(1, 6):
                    if f'C{i}' in col:
                        self._tyre_life_col_map[i] = col
                        break

    def _initialize_first_lap_variables(self, driver: Driver):
        """Set first lap specific variables."""
        if 'FirstLap_pos' in driver.lap_state:
            driver.lap_state['FirstLap_pos'] = float(driver.initial_pos)
        if 'DRS' in driver.lap_state:
            driver.lap_state['DRS'] = 0.0

    def _update_sc_vsc(self) -> bool:
        """
        Update SC/VSC status. Returns True if yellow flag active.

        State transition (global):
        S_SC <- 4 * 1_{SC begins} + max(S_SC - 1, 0)
        S_VSC <- 2 * 1_{VSC begins} + max(S_VSC - 1, 0)
        """
        # Evaluation mode: use predetermined SC/VSC laps.
        # laps_vsc/laps_sc list ALL active laps (not just start laps), so only
        # activate on the FIRST lap of each event to avoid resetting the duration
        # counter on every consecutive lap, which would extend SC/VSC beyond reality.
        if self.eval:
            prev_lap = self.lap_number - 1
            if self.lap_number in self.laps_vsc:
                if prev_lap not in self.laps_vsc:
                    self._activate_vsc()   # first lap of VSC event
                else:
                    self.vsc = 1           # mid-event: keep active, no duration reset
            elif self.lap_number in self.laps_sc:
                if prev_lap not in self.laps_sc:
                    self._activate_sc()    # first lap of SC event
                else:
                    self.sc = 1            # mid-event: keep active, no duration reset
        elif self.vsc == 0 and self.sc == 0:
            # Sample new SC/VSC events
            vsc_triggered = self.np_random.random() < self.lap_vsc_prob
            sc_triggered = self.np_random.random() < self.lap_sc_prob

            if vsc_triggered and sc_triggered and self.race_vsc_prob > 0 and self.race_sc_prob > 0:
                # Both triggered - choose based on relative probability
                vsc_share = self.race_vsc_prob / (self.race_vsc_prob + self.race_sc_prob)
                if self.np_random.random() < vsc_share:
                    self._activate_vsc()
                else:
                    self._activate_sc()
            elif vsc_triggered:
                self._activate_vsc()
            elif sc_triggered:
                self._activate_sc()

        return bool(self.vsc or self.sc) and self.yf_enabled

    def _activate_vsc(self):
        """Activate Virtual Safety Car."""
        self.vsc = 1
        self.laps_left_vsc = VSC_DURATION

    def _activate_sc(self):
        """Activate Safety Car."""
        self.sc = 1
        self.laps_left_sc = SC_DURATION

    def _simulate_rival_pits(self, is_yf: bool) -> np.ndarray:
        """
        Determine pit stop actions for all rivals using VECTORIZED logit model.

        Returns array of compounds (0 = no pit, >0 = pit to that compound).

        Constraints:
        - No pitting in first 5 laps
        - Minimum stint length of 8 laps (unless SC/VSC is active)
        - No pitting in last 7 laps during normal racing
        - During YF, only pit on first lap of SC/VSC period

        OPTIMIZED: Uses batch predictions for all 19 rivals in ONE DataFrame.
        """
        num_rivals = len(self.rival_drivers)
        laps_left = self.n_laps - self.lap_number

        # Evaluation mode: use predetermined strategies
        if self.eval and self.rival_actions_dict:
            return self._get_real_rival_actions()

        # No pitting in first 5 laps (matches training data filter in rivals_logit.py)
        if self.lap_number <= 5:
            return np.zeros(num_rivals, dtype=np.int64)

        # During YF, only allow pitting on first lap of the safety car period
        if is_yf:
            is_first_lap_vsc = (self.vsc > 0 and self.laps_left_vsc == VSC_DURATION)
            is_first_lap_sc = (self.sc > 0 and self.laps_left_sc == SC_DURATION)
            if not (is_first_lap_vsc or is_first_lap_sc):
                return np.zeros(num_rivals, dtype=np.int64)

        # No pitting in last 7 laps during normal racing
        if not is_yf and laps_left <= 7:
            return np.zeros(num_rivals, dtype=np.int64)

        # === VECTORIZED PIT STOP DECISION ===
        rival_actions = np.zeros(num_rivals, dtype=np.int64)
        is_sc = bool(self.sc > 0)
        is_vsc = bool(self.vsc > 0)

        # Build mask for eligible drivers (stint length constraint)
        eligible_mask = np.ones(num_rivals, dtype=bool)
        if not is_yf:
            for i, driver in enumerate(self.rival_drivers):
                if driver.tyrelife < MIN_STINT_LENGTH:
                    eligible_mask[i] = False

        # If no eligible drivers, return zeros
        if not np.any(eligible_mask):
            return rival_actions

        # Get pit probabilities for all eligible drivers in batch
        if self.rival_pitstop_logit is not None:
            pit_probs = self._get_rival_pit_probs_batch(is_sc, is_vsc)

            # Stochastic decisions using vectorized random
            random_thresholds = self.np_random.random(num_rivals)
            pit_decisions = (pit_probs > random_thresholds) & eligible_mask

            # Get compound choices for drivers who decided to pit
            for i, should_pit in enumerate(pit_decisions):
                if should_pit:
                    compound = self._predict_compound_choice(self.rival_drivers[i], laps_left)
                    rival_actions[i] = compound
        else:
            # Fallback to original loop if no model
            for i, driver in enumerate(self.rival_drivers):
                if not eligible_mask[i]:
                    continue
                if self._should_rival_pit(driver, laps_left, is_sc=is_sc, is_vsc=is_vsc):
                    compound = self._predict_compound_choice(driver, laps_left)
                    rival_actions[i] = compound

        return self._ensure_compound_change(rival_actions)

    def _get_rival_pit_probs_batch(self, is_sc: bool, is_vsc: bool) -> np.ndarray:
        """
        Get pit stop probabilities for ALL rivals in ONE batch prediction.

        OPTIMIZED: Creates ONE DataFrame with 19 rows instead of 19 individual DataFrames.
        Speedup: ~15-20x compared to sequential predictions.

        Returns:
            np.ndarray: Pit probabilities for all 19 rivals
        """
        num_rivals = len(self.rival_drivers)

        # Pre-allocate arrays for all features
        sc_vals = np.full(num_rivals, 1.0 if is_sc else 0.0, dtype=np.float32)
        vsc_vals = np.full(num_rivals, 1.0 if is_vsc else 0.0, dtype=np.float32)
        used_two = np.array([1.0 if d.has_changed_compound else 0.0 for d in self.rival_drivers], dtype=np.float32)
        early_laps = np.full(num_rivals, 1.0 if self.lap_number <= 10 else 0.0, dtype=np.float32)
        late_laps = np.full(num_rivals, 1.0 if self.lap_number >= (self.n_laps - 10) else 0.0, dtype=np.float32)
        tyre_life = np.array([float(d.tyrelife) for d in self.rival_drivers], dtype=np.float32)
        compounds = [f'C{d.compound}' for d in self.rival_drivers]
        positions = np.array([float(d.position) for d in self.rival_drivers], dtype=np.float32)

        # Build ONE DataFrame with all rivals
        features_df = pd.DataFrame({
            'SC': sc_vals,
            'VSC': vsc_vals,
            'used_two_compounds': used_two,
            'EarlyLaps': early_laps,
            'LateLaps': late_laps,
            'TyreLife': tyre_life,
            'Compound_Detail': compounds,
            'Position': positions,
            'GP': [self.gp] * num_rivals
        })

        # ONE batch prediction for all rivals
        try:
            probs = self.rival_pitstop_logit.predict(features_df)
            return np.array(probs, dtype=np.float32)
        except Exception:
            # Fallback: return low probability for all
            return np.full(num_rivals, 0.1, dtype=np.float32)

    def _should_rival_pit(self, driver: Driver, laps_left: int,
                          is_sc: bool = False, is_vsc: bool = False) -> bool:
        """
        Determine if rival should pit using logit model.

        Uses the formula: PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps +
                          TyreLife:C(Compound_Detail) + Position + GP
        """
        if self.rival_pitstop_logit is None:
            return False

        # Build features DataFrame for prediction
        features = pd.DataFrame([{
            'SC': 1.0 if is_sc else 0.0,
            'VSC': 1.0 if is_vsc else 0.0,
            'used_two_compounds': 1.0 if driver.has_changed_compound else 0.0,
            'EarlyLaps': 1.0 if self.lap_number <= 10 else 0.0,
            'LateLaps': 1.0 if self.lap_number >= (self.n_laps - 10) else 0.0,
            'TyreLife': float(driver.tyrelife),
            'Compound_Detail': f'C{driver.compound}',
            'Position': float(driver.position),
            'GP': self.gp
        }])

        # Predict probability
        try:
            prob = self.rival_pitstop_logit.predict(features)[0]
        except Exception:
            prob = 0.1  # Fallback probability

        # Sample from probability (stochastic decision)
        return self.np_random.random() < prob

    def _predict_compound_choice(self, driver: Driver, laps_left: int) -> int:
        """
        Choose compound using conditional logit with softmax sampling.

        Uses the formula: chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP
        """
        if self.rival_compound_logit is None:
            # Fallback: random choice from available compounds
            return self.np_random.choice(self.available_compounds)

        # Build features for each alternative compound
        rows = []
        for compound in self.gp_compounds:
            not_change = 1.0 if (compound == f'C{driver.compound}' and
                                  not driver.has_changed_compound) else 0.0
            rows.append({
                'alt': compound,
                'LapsLeft': float(laps_left),
                'not_change_compound': not_change,
                'GP': self.gp
            })

        df = pd.DataFrame(rows)

        try:
            # Get predicted probabilities from logit model
            probs = self.rival_compound_logit.predict(df)

            # Normalize to get valid probability distribution
            probs = np.array(probs)
            probs = probs / probs.sum()

            # Sample compound based on probabilities
            chosen = self.np_random.choice(self.gp_compounds, p=probs)
            return int(chosen[1:])  # 'C3' -> 3
        except Exception:
            # Fallback: random choice
            return self.np_random.choice(self.available_compounds)

    def _ensure_compound_change(self, rival_actions: np.ndarray) -> np.ndarray:
        """Ensure rivals change compound when pitting (if haven't changed yet)."""
        for i, action in enumerate(rival_actions):
            if action > 0 and action == self.rival_drivers[i].compound:
                if not self.rival_drivers[i].has_changed_compound:
                    available_options = [c for c in self.available_compounds if c != action]
                    if available_options:
                        rival_actions[i] = self.np_random.choice(available_options)
        return rival_actions

    def _get_real_rival_actions(self) -> np.ndarray:
        """Get predetermined rival actions for evaluation mode."""
        num_rivals = len(self.rival_drivers)
        rival_actions = np.zeros(num_rivals, dtype=np.int64)

        if not self.rival_actions_dict:
            return rival_actions

        for i, rival in enumerate(self.rival_drivers):
            if rival.name in self.rival_actions_dict:
                strategy = self.rival_actions_dict[rival.name]
                if self.lap_number in strategy:
                    new_compound = strategy[self.lap_number]
                    if new_compound == -1:
                        rival.is_dnf = True
                    elif new_compound > 0 and new_compound in self.available_compounds:
                        rival_actions[i] = new_compound

        return rival_actions

    def _update_all_lap_states(self, agent_action: int, rival_actions: np.ndarray, is_yf: bool):
        """Update lap states for all drivers. OPTIMIZED with O(1) column lookups."""
        intervals = self._calculate_intervals()

        # Agent - pass O(1) lookup maps for maximum performance
        agent_intervals = intervals[self.agent_driver.position - 1] if self.agent_driver.position <= len(intervals) else [100, 100]
        self.agent_driver.lap_state = update_lap_state_v3(
            self.agent_driver.lap_state,
            agent_action,
            self.agent_driver.prev_action,
            self.agent_driver.position,
            agent_intervals,
            is_yf,
            self._compound_cols,
            self._tyre_life_cols,
            self._compound_col_map,
            self._tyre_life_col_map
        )

        # Rivals - pass O(1) lookup maps for maximum performance
        for i, driver in enumerate(self.rival_drivers):
            driver_intervals = intervals[driver.position - 1] if driver.position <= len(intervals) else [100, 100]
            driver.lap_state = update_lap_state_v3(
                driver.lap_state,
                rival_actions[i],
                driver.prev_action,
                driver.position,
                driver_intervals,
                is_yf,
                self._compound_cols,
                self._tyre_life_cols,
                self._compound_col_map,
                self._tyre_life_col_map
            )

    def _calculate_intervals(self) -> List[List[float]]:
        """Calculate intervals between drivers."""
        sorted_drivers = sorted(self.drivers, key=lambda d: d.position)
        n = len(sorted_drivers)
        intervals = []

        for i, driver in enumerate(sorted_drivers):
            is_first = (i == 0)
            is_last = (i == n - 1)

            gap_ahead = 100 if is_first else driver.total_time - sorted_drivers[i - 1].total_time
            gap_behind = 100 if is_last else sorted_drivers[i + 1].total_time - driver.total_time

            intervals.append([gap_ahead, gap_behind])

        return intervals

    def _simulate_lap_times(self, is_yf: bool) -> np.ndarray:
        """
        Calculate lap times for all drivers using FastLinearPredictor.

        Args:
            is_yf: Whether yellow flag conditions are active

        Returns:
            Array of lap times for all drivers (ordered: agent first, then rivals)
        """
        all_drivers = self._ordered_drivers
        laptimes = np.zeros(len(all_drivers))

        for i, driver in enumerate(all_drivers):
            if is_yf and self.fast_predictor_yf is not None:
                # Yellow flag model
                features = build_yf_features(
                    driver.lap_state,
                    is_sc=(self.sc > 0),
                    is_vsc=(self.vsc > 0),
                    is_first_yf_lap=(
                        (self.vsc > 0 and self.laps_left_vsc == VSC_DURATION) or
                        (self.sc > 0 and self.laps_left_sc == SC_DURATION)
                    )
                )
                # Check if it's FastLinearPredictor or statsmodels result
                if hasattr(self.fast_predictor_yf, 'predict_single'):
                    laptime = self.fast_predictor_yf.predict_single(features)
                else:
                    # It's a statsmodels result, use predict with DataFrame
                    # Include both naming conventions (old: SC_start/VSC_start,
                    # new: SC_firstlap/VSC_firstlap); the filter below keeps
                    # only columns that the model actually uses.
                    sc_first = features.get('SC_firstlap', 0.0)
                    vsc_first = features.get('VSC_firstlap', 0.0)
                    features_mapped = {
                        'Intercept': 1.0,
                        'PitIn': features.get('PitIn', 0.0),
                        'PitOut': features.get('PitOut', 0.0),
                        'SC': features.get('SC', 0.0),
                        'VSC': features.get('VSC', 0.0),
                        'SC_start': sc_first,
                        'VSC_start': vsc_first,
                        'SC_firstlap': sc_first,
                        'VSC_firstlap': vsc_first,
                    }
                    # Only use columns that exist in the model
                    model_params = self.fast_predictor_yf.params.index.tolist()
                    features_filtered = {k: v for k, v in features_mapped.items() if k in model_params}
                    features_df = pd.DataFrame([features_filtered])
                    laptime = self.fast_predictor_yf.predict(features_df)[0]
            else:
                # Clear track model with OLS prediction variance:
                # y ~ N(y_hat, sigma^2 * [1 + X0^T (X^T X)^{-1} X0])
                if not self.deterministic and hasattr(self.fast_predictor_clear, 'predict_single_with_std'):
                    laptime, pred_std = self.fast_predictor_clear.predict_single_with_std(driver.lap_state)
                    laptime = self.np_random.normal(laptime, pred_std)
                else:
                    laptime = self.fast_predictor_clear.predict_single(driver.lap_state)

            laptimes[i] = laptime
            driver.predicted_laptime = laptime

        # Store agent laptime for reward calculation
        self._agent_last_laptime = laptimes[0]

        return laptimes

    def _resolve_overtakes(self, is_yf: bool = False):
        """
        Simulate overtakes following Algorithm 2 from paper (OPTIMIZED IMPLEMENTATION).

        OPTIMIZATION: Pre-computes ALL possible overtake probabilities in ONE batch call.
        The gaps change during execution due to swaps, but we batch all initial τ predictions
        and use a lookup table during the algorithm execution.

        Algorithm: System evolution for overtakes
        For h = 2 to 20:
            For k = h down to 2:
                X_Δ2^(k,k-1) ← gap at lap start between d^(k) and d^(k-1)
                X_PredGap ← X_Δ2^(k,k-1) + ŷ_d(k) − ŷ_d(k-1)

                If Clear track: Overtake ← Uniform(0,1) < τ (stochastic)
                If SC/VSC: Overtake ← X_PredGap < 0 (deterministic)

                If Overtake:
                    ŷ_d(k)   ← max(ŷ_d(k),   ŷ_d(k-2) − X_Δ2^(k,k-2) + t_min_gap)
                    ŷ_d(k-1) ← max(ŷ_d(k-1), ŷ_d(k)   + X_Δ2^(k,k-1) + t_min_gap)
                    Swap d^(k) and d^(k-1)
                Else:
                    ŷ_d(k) ← max(ŷ_d(k), ŷ_d(k-1) − X_Δ2^(k,k-1) + t_min_gap)
                    break

        Args:
            is_yf: Whether yellow flag (SC/VSC) is active
        """
        # Sort drivers by position at start of lap
        sorted_drivers = sorted(self.drivers, key=lambda d: d.position)
        n_drivers = len(sorted_drivers)

        # Store original cumulative times at lap start (before this lap's times)
        original_total_times = {d.name: d.total_time for d in sorted_drivers}

        # Working copy of predicted lap times (will be modified for gap enforcement)
        pred_laptimes = {d.name: d.predicted_laptime for d in sorted_drivers}

        # === BATCH PRE-COMPUTATION OF OVERTAKE PROBABILITIES ===
        # On clear track, pre-compute τ for ALL possible adjacent pairs
        # This avoids O(n²) individual predict() calls
        tau_lookup = {}
        if not is_yf and self.logit_overtake is not None:
            tau_lookup = self._compute_overtake_probs_batch(sorted_drivers, original_total_times, pred_laptimes)

        # Pre-generate random thresholds for all possible overtake attempts
        # Max overtakes per step: roughly n*(n-1)/2 = 190 for 20 drivers
        random_thresholds = self.np_random.random(200)
        random_idx = 0

        # Outer loop: for h = 2 to 20 (1-indexed), so h = 1 to 19 (0-indexed)
        for h in range(1, n_drivers):
            k = h
            while k >= 1:
                driver_k = sorted_drivers[k]      # Attacker (behind)
                driver_k_1 = sorted_drivers[k-1]  # Defender (ahead)

                # X_Δ2^(k,k-1): gap at lap start between d^(k) and d^(k-1)
                gap_k_to_k1 = original_total_times[driver_k.name] - original_total_times[driver_k_1.name]

                # X_PredGap = X_Δ2^(k,k-1) + ŷ_d(k) − ŷ_d(k-1)
                pred_gap = gap_k_to_k1 + pred_laptimes[driver_k.name] - pred_laptimes[driver_k_1.name]

                # Determine if overtake occurs
                if is_yf:
                    # SC/VSC: Deterministic overtake based on predicted gap
                    overtake = pred_gap < 0
                else:
                    # Clear track: Use pre-computed τ from batch lookup
                    # Key is the pair of driver names (attacker, defender)
                    pair_key = (driver_k.name, driver_k_1.name)
                    if pair_key in tau_lookup:
                        tau = tau_lookup[pair_key]
                    elif self.logit_overtake is not None:
                        # Fallback for pairs not in initial lookup (due to swaps)
                        # Use quick approximation based on pred_gap
                        tau = self._quick_tau_estimate(pred_gap)
                    else:
                        tau = 0.7 if pred_gap < -0.5 else 0.3

                    # CRITICAL: Overtake ← Uniform(0,1) < τ (STOCHASTIC)
                    overtake = random_thresholds[random_idx] < tau
                    random_idx += 1

                if overtake:
                    # === OVERTAKE OCCURS ===
                    # ŷ_d(k) ← max{ŷ_d(k), ŷ_d(k-2) − X_Δ2^(k,k-2) + t_min_gap}
                    # When k < 2 (overtaking into P1), no car ahead → no constraint
                    if k >= 2:
                        driver_k_2 = sorted_drivers[k-2]
                        gap_k_to_k2 = original_total_times[driver_k.name] - original_total_times[driver_k_2.name]
                        pred_laptimes[driver_k.name] = max(
                            pred_laptimes[driver_k.name],
                            pred_laptimes[driver_k_2.name] - gap_k_to_k2 + MIN_GAP
                        )

                    # ŷ_d(k-1) ← max{ŷ_d(k-1), ŷ_d(k) + X_Δ2^(k,k-1) + t_min_gap}
                    pred_laptimes[driver_k_1.name] = max(
                        pred_laptimes[driver_k_1.name],
                        pred_laptimes[driver_k.name] + gap_k_to_k1 + MIN_GAP
                    )

                    # Swap d^(k) and d^(k-1) in sorted array
                    sorted_drivers[k], sorted_drivers[k-1] = sorted_drivers[k-1], sorted_drivers[k]
                    k -= 1
                else:
                    # === NO OVERTAKE ===
                    pred_laptimes[driver_k.name] = max(
                        pred_laptimes[driver_k.name],
                        pred_laptimes[driver_k_1.name] - gap_k_to_k1 + MIN_GAP
                    )
                    break

        # Apply adjusted predicted lap times to total times
        for driver in self.drivers:
            original_time_before_lap = original_total_times[driver.name]
            driver.total_time = original_time_before_lap + pred_laptimes[driver.name]
            driver.predicted_laptime = pred_laptimes[driver.name]

    def _compute_overtake_probs_batch(
        self,
        sorted_drivers: list,
        original_total_times: dict,
        pred_laptimes: dict
    ) -> dict:
        """
        Batch compute overtake probabilities for all adjacent driver pairs.

        OPTIMIZED: Creates ONE DataFrame with all pairs and calls predict() ONCE.
        Speedup: ~5-10x compared to O(n²) individual predictions.

        Returns:
            dict: {(attacker_name, defender_name): tau_probability}
        """
        n_drivers = len(sorted_drivers)

        # Collect all adjacent pairs and their predicted gaps
        pairs = []
        pred_gaps = []

        for k in range(1, n_drivers):
            driver_k = sorted_drivers[k]
            driver_k_1 = sorted_drivers[k-1]

            gap_k_to_k1 = original_total_times[driver_k.name] - original_total_times[driver_k_1.name]
            pred_gap = gap_k_to_k1 + pred_laptimes[driver_k.name] - pred_laptimes[driver_k_1.name]

            pairs.append((driver_k.name, driver_k_1.name))
            pred_gaps.append(pred_gap)

        if not pairs:
            return {}

        # Build ONE DataFrame with all pairs
        features_df = pd.DataFrame({
            'const': np.ones(len(pairs), dtype=np.float32),
            'delta_total': np.array(pred_gaps, dtype=np.float32)
        })

        # ONE batch prediction
        try:
            taus = self.logit_overtake.predict(features_df)
            return dict(zip(pairs, taus))
        except Exception:
            # Fallback: return estimated taus
            return {pair: (0.7 if gap < -0.5 else 0.3) for pair, gap in zip(pairs, pred_gaps)}

    def _quick_tau_estimate(self, pred_gap: float) -> float:
        """Quick τ estimate when batch lookup misses (due to position swaps)."""
        # Simple logistic approximation based on typical overtake model
        # τ ≈ 1 / (1 + exp(β * pred_gap)) where β ≈ 1.5
        # Clip to avoid overflow in exp()
        clamped_gap = np.clip(1.5 * pred_gap, -500, 500)
        return 1.0 / (1.0 + np.exp(clamped_gap))

    def _update_positions_and_intervals(self):
        """
        Update positions based on total race time.

        State transition: S_Position[d] <- rank(cumulative_time)
        """
        self.drivers.sort(key=lambda d: d.total_time)
        for i, driver in enumerate(self.drivers):
            driver.position = i + 1

    def _update_driver_states(self, agent_action: int, rival_actions: np.ndarray, is_yf: bool):
        """
        Update driver states after lap completion.

        Mandatory state transitions (per driver d with action a):
        S_Tire[d]      <- S_Tire[d] * 1_{a=0} + a * 1_{a!=0}
        S_TireLife[d]  <- (S_TireLife[d] + 1_{SC+VSC=0}) * 1_{a=0} + 0 * 1_{a!=0}
        S_Change[d]    <- min(S_Change[d] + 1_{a not in {0, S_Tire[d]}}, 1)
        S_PrevAction[d]<- a
        """
        # Combine actions: agent first, then rivals
        actions = np.concatenate([[agent_action], rival_actions])

        for driver, action in zip(self._ordered_drivers, actions):
            old_compound = driver.compound

            if action == 0:
                # No pit stop - increment tyre life unless under yellow flag
                if not is_yf:
                    driver.tyrelife += 1
                # S_Tire[d] stays the same
            else:
                # Pit stop - reset tyre life and update compound
                driver.tyrelife = 0
                driver.compound = action

                # S_Change[d] <- min(S_Change[d] + 1_{a not in {0, S_Tire[d]}}, 1)
                if old_compound != 0 and action != old_compound:
                    driver.has_changed_compound = True

            # S_PrevAction[d] <- a
            driver.prev_action = action

    def _update_safety_car_duration(self):
        """
        Update SC/VSC countdown.

        State transition:
        S_SC <- max(S_SC - 1, 0)
        S_VSC <- max(S_VSC - 1, 0)
        """
        if self.vsc > 0:
            self.laps_left_vsc = max(0, self.laps_left_vsc - 1)
            if self.laps_left_vsc == 0:
                self.vsc = 0
        if self.sc > 0:
            self.laps_left_sc = max(0, self.laps_left_sc - 1)
            if self.laps_left_sc == 0:
                self.sc = 0

    def _calculate_reward(self, action: int, done: bool) -> float:
        """
        Calculate reward for the agent based on self.reward_type.

        Reward types:
        - 'mix': Negative lap time per step + terminal points & position bonuses
        - 'time': Centered per-lap time (laptime - baseline), no terminal bonus
        - 'position': Terminal -final_position only (no per-step lap time)
        - 'points': Terminal F1 points only (no per-step lap time)

        All types apply the same penalties (disqualification, pit last lap,
        early pit, consecutive pit, invalid) scaled to the reward magnitude.

        F1 REGULATION: Drivers must use at least 2 different tire compounds.
        Failure to do so results in DISQUALIFICATION.
        """
        reward = 0.0
        rt = self.reward_type

        # --- Per-step cost ---
        if rt == 'mix' and self._agent_last_laptime > 0:
            reward += -self._agent_last_laptime
        elif rt == 'time' and self._agent_last_laptime > 0:
            reward += -(self._agent_last_laptime - self._baseline_laptime)

        # Map action to compound for penalty checks
        if action == 0:
            pit_compound = 0
        elif 1 <= action <= self._n_compounds:
            pit_compound = self.available_compounds[action - 1]
        else:
            pit_compound = 0

        # Consecutive pit penalty (all reward types)
        if action > 0 and self.agent_driver.prev_action > 0:
            reward += self._penalty_consecutive_pit

        # Early pit penalty, laps 1-5 (all reward types)
        if pit_compound > 0 and (self.lap_number - 1) in self._early_laps_set:
            reward += self._penalty_early_pit

        if done:
            final_pos = self.agent_driver.position

            # F1 REGULATION CHECK: Must use at least 2 different compounds
            if not self.agent_driver.has_changed_compound:
                reward += self._penalty_disqualification
                if self.verbose:
                    print(f"DISQUALIFIED: Agent did not change tire compound (F1 regulation)")
            else:
                # Pit on final lap penalty (all reward types)
                if pit_compound > 0:
                    reward += self._penalty_pit_last_lap

                # --- Terminal bonus (reward-type specific) ---
                if rt == 'mix':
                    # Points for top 10
                    if final_pos <= 10:
                        reward += POINTS_SYSTEM[final_pos - 1] * self._points_multiplier
                    # Position gradient for all positions
                    reward += (21 - final_pos) * self._position_multiplier

                elif rt == 'position':
                    # Negative position x5: P1=-5, P20=-100
                    reward += -float(final_pos) * 5

                elif rt == 'points':
                    # Raw F1 points: P1=25, ..., P10=1, P11+=0
                    reward += float(POINTS_SYSTEM[final_pos - 1])

                # 'time': no terminal bonus (uses per-step centered lap time)

        return reward

    def _update_obs_cache(self):
        """Update the observation cache in-place."""
        drivers = self._ordered_drivers
        agent_time = self.agent_driver.total_time if self.agent_driver else 0.0

        # Update arrays
        for i, d in enumerate(drivers):
            self._obs_cache["compounds"][i] = d.compound
            self._obs_cache["tyrelifes"][i] = d.tyrelife
            self._obs_cache["change_compounds"][i] = int(d.has_changed_compound)
            self._obs_cache["positions"][i] = d.position

        # Time differences to agent (rivals only)
        for i, d in enumerate(self.rival_drivers):
            self._obs_cache["time_diff_to_agent"][i] = d.total_time - agent_time

        # Scalars
        self._obs_cache["lap_number"] = self.lap_number
        self._obs_cache["laps_left"] = self.laps_left
        self._obs_cache["prev_action"] = self.agent_driver.prev_action if self.agent_driver else 0
        self._obs_cache["vsc"] = self.laps_left_vsc if self.vsc > 0 else 0
        self._obs_cache["sc"] = self.laps_left_sc if self.sc > 0 else 0

    @property
    def _ordered_drivers(self) -> List[Driver]:
        """Returns drivers in consistent order: agent first, then rivals."""
        return [self.agent_driver] + self.rival_drivers

    def set_evaluation_mode(
        self,
        eval_mode: bool = True,
        laps_vsc: List[int] = None,
        laps_sc: List[int] = None,
        initial_positions: List[int] = None,
        rival_actions: Dict = None
    ):
        """Configure environment for evaluation with predetermined events."""
        self.eval = eval_mode
        self.laps_vsc = laps_vsc or []
        self.laps_sc = laps_sc or []
        if initial_positions:
            self.initial_positions = initial_positions
        self.rival_actions_dict = rival_actions or {}

    @property
    def agent_position(self) -> int:
        """Returns agent's current position."""
        return self.agent_driver.position if self.agent_driver else 0

    @property
    def agent_total_time(self) -> float:
        """Returns agent's total race time."""
        return self.agent_driver.total_time if self.agent_driver else 0.0

    @property
    def race_classification(self) -> List[Dict[str, Any]]:
        """Returns current race classification."""
        sorted_drivers = sorted(self.drivers, key=lambda d: d.total_time)
        return [
            {
                'position': i + 1,
                'name': d.name,
                'total_time': d.total_time,
                'compound': d.compound,
                'tyrelife': d.tyrelife,
                'has_changed_compound': d.has_changed_compound
            }
            for i, d in enumerate(sorted_drivers)
        ]

    def _log_init_info(self):
        """Log initialization information."""
        print(f"\n{'='*60}")
        print(f"F1 Multi-Driver Environment Initialized: {self.gp}")
        print(f"{'='*60}")
        print(f"Laps: {self.n_laps}")
        print(f"Compounds: {self.available_compounds}")
        print(f"Agent: {self.agent_driver_name}")
        print(f"VSC/SC prob: {self.race_vsc_prob:.2f}/{self.race_sc_prob:.2f}")
        print(f"Deterministic: {self.deterministic}")
        print(f"{'='*60}\n")


# === FACTORY FUNCTIONS ===

def create_f1_env_all_drivers(
    gp: str,
    driver: str,
    deterministic: bool = False,
    yf_enabled: bool = True,
    initial_positions: List[int] = None,
    models_path: str = None,
    loader: ModelLoader = None,
    verbose: bool = False,
    reward_type: str = 'mix',
) -> F1EnvAllDrivers:
    """
    Create F1 multi-driver environment using the unified ModelLoader.

    This is the recommended way to create environments as it uses the
    organized model structure in Models/simulation/.

    Args:
        gp: Grand Prix name (e.g., 'Belgian Grand Prix')
        driver: Agent driver name (e.g., 'Driver_VER')
        deterministic: If True, no randomness in simulation
        yf_enabled: If True, simulate SC/VSC events
        initial_positions: Fixed starting positions for evaluation
        models_path: Custom path to Models/simulation (optional)
        loader: Pre-initialized ModelLoader (optional, for caching)
        verbose: Print debug information
        reward_type: Reward structure ('mix', 'time', 'position', 'points')

    Returns:
        F1EnvAllDrivers: Configured environment instance

    Example:
        # Simple usage
        env = create_f1_env_all_drivers('Belgian Grand Prix', 'Driver_VER')

        # With caching across multiple environments
        loader = ModelLoader()
        env1 = create_f1_env_all_drivers('Belgian Grand Prix', 'Driver_VER', loader=loader)
        env2 = create_f1_env_all_drivers('Hungarian Grand Prix', 'Driver_VER', loader=loader)
    """
    # Initialize loader if not provided
    if loader is None:
        if models_path:
            loader = ModelLoader(models_path)
        else:
            loader = ModelLoader()

    # Load all models for the GP
    models = loader.load(gp)

    return F1EnvAllDrivers(
        n_laps=models.n_laps,
        available_compounds=models.compounds,
        gp=models.gp_name,
        agent_driver=driver,
        fast_predictor_clear=models.laptime_clear,
        fast_predictor_yf=models.laptime_yf,
        logit_overtake=models.overtake_logit,
        rival_pitstop_logit=models.rival_pitstop_logit,
        rival_compound_logit=models.rival_compound_logit,
        vsc_prob=models.vsc_prob,
        sc_prob=models.sc_prob,
        overtake_threshold=models.overtake_threshold,
        deterministic=deterministic,
        yf_enabled=yf_enabled,
        initial_positions=initial_positions,
        verbose=verbose,
        reward_type=reward_type,
    )


def create_f1_env_from_models(
    models: GPModels,
    driver: str,
    deterministic: bool = False,
    yf_enabled: bool = True,
    initial_positions: List[int] = None,
    verbose: bool = False,
    reward_type: str = 'mix',
) -> F1EnvAllDrivers:
    """
    Create F1 multi-driver environment directly from a GPModels instance.

    Use this when you already have models loaded and want to avoid
    reloading them.

    Args:
        models: GPModels container with all required models
        driver: Agent driver name (e.g., 'Driver_VER')
        deterministic: If True, no randomness
        yf_enabled: If True, simulate SC/VSC
        initial_positions: Fixed starting positions
        verbose: Print debug information
        reward_type: Reward structure ('mix', 'time', 'position', 'points')

    Returns:
        F1EnvAllDrivers: Configured environment instance
    """
    return F1EnvAllDrivers(
        n_laps=models.n_laps,
        available_compounds=models.compounds,
        gp=models.gp_name,
        agent_driver=driver,
        fast_predictor_clear=models.laptime_clear,
        fast_predictor_yf=models.laptime_yf,
        logit_overtake=models.overtake_logit,
        rival_pitstop_logit=models.rival_pitstop_logit,
        rival_compound_logit=models.rival_compound_logit,
        vsc_prob=models.vsc_prob,
        sc_prob=models.sc_prob,
        overtake_threshold=models.overtake_threshold,
        deterministic=deterministic,
        yf_enabled=yf_enabled,
        initial_positions=initial_positions,
        verbose=verbose,
        reward_type=reward_type,
    )


# === ENTRY POINT FOR TESTING ===

if __name__ == "__main__":
    print("Testing F1EnvAllDrivers...")
    print("="*70)

    try:
        # Test with ModelLoader-based factory
        env = create_f1_env_all_drivers('Bahrain Grand Prix', 'Driver_VER', verbose=True)
        print(f"Environment created successfully!")
        print(f"  GP: {env.gp}")
        print(f"  Laps: {env.n_laps}")
        print(f"  Compounds: {env.available_compounds}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")

        # Run a few steps
        obs, _ = env.reset()
        print(f"\nInitial observation:")
        print(f"  Positions: {obs['positions'][:5]}...")
        print(f"  Lap number: {obs['lap_number']}")

        # Choose starting compound (action 1 = softest available)
        obs, reward, done, _, _ = env.step(1)
        print(f"\nAfter choosing compound:")
        print(f"  Agent compound: {env.agent_driver.compound}")
        print(f"  Lap number: {obs['lap_number']}")

        # Run a few more steps
        print("\nRunning 5 laps...")
        for i in range(5):
            obs, reward, done, _, _ = env.step(0)  # No pit
            print(f"  Lap {obs['lap_number']}: Agent pos={env.agent_position}, reward={reward:.2f}")

        # Run full race
        print("\nRunning full race...")
        total_reward = 0.0
        while not done:
            obs, reward, done, _, _ = env.step(0)
            total_reward += reward

        print(f"\nRace complete!")
        print(f"  Final position: {env.agent_position}")
        print(f"  Total time: {env.agent_total_time:.2f}s")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Changed compound: {env.agent_driver.has_changed_compound}")

        print("\n" + "="*70)
        print("Test completed successfully!")
        print("="*70)

    except FileNotFoundError as e:
        print(f"Models not found. Please ensure models exist in Models/simulation/")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
