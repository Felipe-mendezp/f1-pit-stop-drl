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
import tyre_sets


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

# --- Following-gap model (opt-in, 2026-07-31) -------------------------------
# Clamping a denied overtake to the CONSTANT MIN_GAP welds cars into trains at
# exactly 0.4 s. Measured on the Singapore prefix, 23% of all inter-car gaps sat
# at exactly 0.40 s, whereas the real 2024 race has only 3.1% of gaps below
# 0.5 s and a median of 2.09 s (data/session_2024_V2.csv, Interval_front). The
# artefact matters because it turns position-vs-time into a coarse step
# function: after a pit stop the rejoin slot — and hence the final position —
# jumps by 2-3 places for a one-lap change in pit timing.
#
# With `following_gap_scale > 0` the clamp instead uses
#     MIN_GAP + Exponential(mean = following_gap_scale)
# resampled per pair per lap, which reproduces a spread-out field. Default 0.0
# keeps the exact legacy behaviour, so nothing changes unless enabled.
FOLLOWING_GAP_SCALE_DEFAULT = 0.0
# LogitOvertake is trained ONLY on adjacent pairs within 1.0s (see
# logit_overtakes.py). Outside that support the logit must not be
# extrapolated: for larger start-of-lap gaps the pass is deterministic
# (attacker ends the lap ahead => passes, e.g. defender pitted).
OVERTAKE_MAX_GAP = 1.0
# --- Pit-lane pass (opt-in, 2026-08-04) --------------------------------------
# The support rule above keys on the START-OF-LAP gap, so a pair whose defender
# pits THIS lap (pred_gap ~ -10s in-lap / -24s out-lap from PitIn/PitOut) still
# reaches the stochastic branch whenever the attacker starts within 1.0s. That
# extrapolates the logit far outside its training range: at Singapore
# tau(-12s)=0.19, so ~81% of the time the attacker FAILS to pass a car that is
# in the pit lane and the no-overtake clamp welds it (and, chained, the whole
# train) 0.4s behind the pitting car -> the pit stop is effectively free and
# the decision-support outcome flips 4-6 final positions on a one-lap change
# of pit timing (diagnosed 2026-08-04 on the Singapore_HUL bundle).
# With `pit_pass_deterministic = True` any battle involving a driver whose
# current lap carries PitIn or PitOut is resolved deterministically
# (overtake <=> pred_gap < 0), exactly like the out-of-support branch. Default
# False keeps the exact legacy behaviour.
PIT_PASS_DETERMINISTIC_DEFAULT = False
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
    # Tyre sets (opt-in F1_TYRE_SETS, 2026-10-21, see src/tyre_sets.py): sets left {compound: ascending ages},
    # laps of the current stint (= tyrelife without the flag) and age of the set fitted at the last stop
    sets: Optional[Dict[int, List[int]]] = None
    stint_laps: int = 0
    next_set_age: int = 0


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
        'laps_left_vsc', 'laps_left_sc', '_is_first_yf_lap',
        # Calibrated overtake logit (intercept/slope in logit space)
        '_ot_intercept', '_ot_slope',
        # Fast-path rival models (opt-in, see fast_rivals)
        'fast_rivals', '_stop_betas', '_choice_betas',
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
        'eval_initial_compounds',
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
        fast_rivals: bool = False,
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
        self._is_first_yf_lap = False

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
        self.eval_initial_compounds = {}

        # Last laptime for reward calculation
        self._agent_last_laptime = 0.0
        self._baseline_laptime = 0.0

        # Calibrate the overtake logit fallback (intercept/slope)
        self._calibrate_overtake_logit()

        # OPT-IN fast path for rival decisions (~3x faster env steps):
        # extracts the logit coefficients ONCE and evaluates them in pure
        # numpy, instead of building a pandas DataFrame + statsmodels
        # predict() every lap. Numerically validated against the slow path
        # at init; falls back to the slow path on any mismatch.
        self.fast_rivals = bool(fast_rivals)
        self._stop_betas = None
        self._choice_betas = None
        if self.fast_rivals:
            self._calibrate_rival_fast_path()

        # Tyre sets (opt-in F1_TYRE_SETS, 2026-10-21): read here because the observation space depends on it
        self.tyre_sets = tyre_sets.enabled()
        self._no_set_penalty = tyre_sets.no_set_penalty()
        self._no_set_pending = 0.0
        self._real_stint_ages = {}
        # 2026-10-22 (v9r2), opt-in F1_TYRE_REUSE=1: reusable sets (see src/tyre_sets.py) for every driver; the
        # agent's 'tyre_sets' = age / 20 of the set a stop would fit, per compound
        self._reuse = tyre_sets.reuse_enabled()
        self._n_sets_obs = self._n_compounds if self._reuse else 2 * self._n_compounds

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
        if self.tyre_sets:
            self.observation_space.spaces["tyrelifes"] = Box(
                low=0, high=n_laps + 1 + tyre_sets.MAX_AGE, shape=(20,), dtype=np.float32)
            self.observation_space.spaces["tyre_sets"] = Box(
                low=0, high=float(tyre_sets.MAX_AGE), shape=(self._n_sets_obs,), dtype=np.float32)

        # Action space: 0=no pit, 1-3=compounds (hardest to softest)
        self.action_space = Discrete(4)

        if verbose:
            self._log_init_info()

    def _create_obs_cache(self) -> Dict[str, Any]:
        """Create initial observation cache."""
        return {
            "compounds": np.zeros(20, dtype=np.float32),
            "tyrelifes": np.zeros(20, dtype=np.float32),
            "change_compounds": np.zeros(20, dtype=np.int8),
            "positions": np.zeros(20, dtype=np.float32),
            "time_diff_to_agent": np.zeros(19, dtype=np.float32),
            "lap_number": 0,
            "laps_left": self.n_laps,
            "prev_action": 0,
            "vsc": 0,
            "sc": 0,
            **({"tyre_sets": np.zeros(self._n_sets_obs, dtype=np.float32)} if self.tyre_sets else {}),
        }

    def _get_obs(self) -> Dict[str, Any]:
        """
        Return a safe copy of the observation: the cached numpy arrays are
        copied so stored observations are not mutated by later steps.
        """
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self._obs_cache.items()
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._initialize_race_state()
        # ABANDONOS DE RIVALES (opt-in `rival_dnf`, 2026-10-14): con rivales
        # estocasticos, cada rival abandona con prob. p_dnf en la vuelta
        # round(frac * n_laps) (frac de la distribucion empirica 2021-23, ver
        # scripts/rival_dnf_calibration_20261014.py). Antes nunca abandonaban,
        # y en la Fig. 7 (estrategias reales con -1) el rival retirado
        # (time_diff = +500) quedaba fuera de distribucion. Con estrategias
        # reales (eval + rival_actions_dict) no se usa. Se sortea con un
        # generador propio derivado de np_random (una sola extraccion).
        self._rival_dnf_lap = {}
        cal = getattr(self, 'rival_dnf', None)
        if cal and not (self.eval and self.rival_actions_dict):
            g = np.random.default_rng(int(self.np_random.integers(2**62)))
            fr = cal['frac_completed']
            for d in self.rival_drivers:
                if g.random() < cal['p_dnf']:
                    f = fr[int(g.integers(len(fr)))]
                    self._rival_dnf_lap[d.name] = int(min(max(round(f * self.n_laps), 1), self.n_laps - 1))
        self._update_obs_cache()
        # LARGADA SORTEADA (opt-in `start_mix`, 2026-10-25, post-entrenamiento,
        # docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md): con prob.
        # start_mix la vuelta 0 se juega aqui con un compuesto uniforme (accion
        # valida, misma toma de juego) y el episodio empieza en la vuelta 1. Asi
        # la politica ve todas las largadas sin que PPO le atribuya una accion de
        # vuelta 0 que no tomo. Generador propio derivado de np_random (una sola
        # extraccion). Default 0 = sin cambio.
        if getattr(self, 'start_mix', 0.0) > 0:
            g = np.random.default_rng(int(self.np_random.integers(2**62)))
            if g.random() < self.start_mix:
                obs, _, _, _, _ = self._handle_lap_0(1 + int(g.integers(len(self.available_compounds))))
                return obs, {'start_mix': True}
        return self._get_obs(), {}

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

        # --- RULE MASK (opt-in `rule_mask`, 2026-09-01) -------------------
        # Misma lógica que _rule_infeasible del decision support, aplicada al
        # índice de acción (así reward y dinámica ven la acción enmascarada):
        #   (a) pit al compuesto ya montado con la regla de dos compuestos aún
        #       sin cumplir -> no-op (paga el pit sin progresar hacia la regla);
        #   (b) no parar en la última vuelta en que aún se puede cumplir la
        #       regla -> parada forzada a un compuesto distinto (el entorno
        #       hace cumplir la regla, como haría dirección de carrera).
        # Objetivo: quitar el precipicio de descalificación (-5000) del paisaje
        # de optimización, que ancla a PPO/RPPO en el atractor "pit en L1"
        # cuando se sale desde el fondo de la parrilla (Miami).
        #   (c) pit en la VUELTA FINAL -> no-op: en el simulador esa parada
        #       paga solo PitIn (nunca existe la vuelta de salida), un loophole
        #       de ~2.5 posiciones que el piloto v1 (2026-09-01) explotó.
        # Orden: (c) primero; luego (a); luego (b) sobre la acción ya
        # redirigida, para que un pit al mismo compuesto en la última vuelta
        # factible se convierta en la parada forzada legal y no en no-op.
        if getattr(self, 'rule_mask', False):
            _changed = self.agent_driver.has_changed_compound
            _cur = self.agent_driver.compound
            if self.lap_number >= self.n_laps and 1 <= action <= self._n_compounds:
                action = 0
            if (1 <= action <= self._n_compounds and (not _changed)
                    and self.available_compounds[action - 1] == _cur):
                action = 0
            if (action == 0 and (not _changed)
                    and self.lap_number == self.n_laps - 1):
                _opts = [a for a in range(1, self._n_compounds + 1)
                         if self.available_compounds[a - 1] != _cur]
                if self.tyre_sets:  # forced stop to a compound with sets left (if any)
                    _opts = [a for a in _opts if self._has_set(self.agent_driver, self.available_compounds[a - 1])] \
                        or _opts
                action = _opts[0]

        # Map action to compound (for laps > 0)
        if action == 0:
            agent_action = 0  # No pit
        elif 1 <= action <= self._n_compounds:
            agent_action = self.available_compounds[action - 1]
        else:
            # Invalid action during race
            self._update_obs_cache()
            return self._get_obs(), self._penalty_invalid, False, False, {}

        # Tyre sets (opt-in): a stop to a compound with no set left does not happen and is penalised
        self._no_set_pending = 0.0
        if self.tyre_sets and agent_action != 0 and not self._has_set(self.agent_driver, agent_action):
            agent_action = 0
            self._no_set_pending = self._no_set_penalty

        # === STRICT PER-LAP EVOLUTION ORDER ===

        # Snapshot the agent's previous action BEFORE step 7 overwrites it;
        # the consecutive-pit penalty must compare against the prior lap.
        agent_prev_action = self.agent_driver.prev_action

        # Reward relativo (opt-in `reward_relative`, 2026-09-04): snapshot de los
        # tiempos acumulados para medir el delta de vuelta REAL de cada piloto
        # (post-adelantamientos, post-bunching) y restar la mediana del pelotón.
        if getattr(self, 'reward_relative', False):
            self._tt_before = {d.name: d.total_time for d in self._ordered_drivers}

        # 1. Update safety car status
        # SC REVEAL (opt-in `sc_reveal`, 2026-10-07): el SC/VSC de esta vuelta
        # ya se sorteó/leyó al final del paso anterior (_reveal_next_lap_yf),
        # así que la observación con que el agente decidió ya lo mostraba,
        # igual que los rivales (que deciden tras el sorteo). La secuencia de
        # llamadas al RNG es la misma que sin el flag (mismos eventos).
        if getattr(self, 'sc_reveal', False):
            is_yf = self._revealed_is_yf
        else:
            is_yf = self._update_sc_vsc()

        # --- SC BUNCHING (opt-in `sc_bunching`, 2026-08-28) -----------------
        # Índice de vuelta DENTRO del período SC vigente (1 = primera vuelta
        # con SC). Con el mecanismo activo, el período SC se modela así:
        #   v1: se puede parar (única vuelta con pit permitido); laptime con
        #       coef SC (sin SC_firstlap — ver _simulate_lap_times)
        #   v2: pit prohibido; coef SC
        #   v3+: pelotón AGLOMERADO detrás del líder (_apply_sc_bunching);
        #        pit prohibido; el líder avanza con coef SC y el resto queda
        #        en fila a (pos-1)·sc_bunch_delta -> los gaps se borran.
        # Los rivales ya tenían la restricción "solo 1a vuelta de YF"
        # (_simulate_rival_pits); aquí se extiende al agente (no-op sin
        # castigo). VSC queda INTACTO (en la realidad no aglomera).
        # (con sc_reveal el índice ya se actualizó en _reveal_next_lap_yf)
        if not getattr(self, 'sc_reveal', False):
            if self.sc > 0:
                self._sc_lap_idx = getattr(self, '_sc_lap_idx', 0) + 1
            else:
                self._sc_lap_idx = 0
        if (getattr(self, 'sc_bunching', False) and self.sc > 0
                and self._sc_lap_idx >= 2 and agent_action != 0):
            agent_action = 0

        # Abandonos sorteados en reset (opt-in `rival_dnf`): mismo momento que
        # el -1 de las estrategias reales (_get_real_rival_actions, paso 2).
        if getattr(self, '_rival_dnf_lap', None):
            for d in self.rival_drivers:
                if self._rival_dnf_lap.get(d.name) == self.lap_number:
                    d.is_dnf = True

        # 2. Simulate rival pit decisions
        rival_actions = self._simulate_rival_pits(is_yf)

        # 3. Update all lap states
        self._update_all_lap_states(agent_action, rival_actions, is_yf)

        # 4. Simulate lap times (preliminary ŷ_d for all drivers)
        self._simulate_lap_times(is_yf)

        # 5. Resolve overtakes and apply lap times to cumulative total
        #    (o aglomeración SC, opt-in: v2 compacta gaps ×0.5, v3+ fila completa)
        if (getattr(self, 'sc_bunching', False) and self.sc > 0
                and getattr(self, '_sc_lap_idx', 0) >= 3):
            self._apply_sc_bunching()
        else:
            self._resolve_overtakes(is_yf)
            if (getattr(self, 'sc_bunching', False) and self.sc > 0
                    and getattr(self, '_sc_lap_idx', 0) == 2):
                # Compactación PROGRESIVA (2026-08-28, opción elegida por el
                # autor): en la 2a vuelta de SC el pelotón ya se está
                # recogiendo — los gaps al líder se escalan ×0.5 (tras aplicar
                # los tiempos normales, incluidas las out-laps de quienes
                # pararon en la v1). Así el pit bajo SC no paga su pérdida
                # completa contra un pelotón estirado antes de que la fila de
                # la v3 congele el orden.
                self._compact_sc_gaps(0.5)

        # Reward must see the lap time actually applied to the cumulative
        # total (post traffic/overtake adjustments), not the preliminary ŷ.
        self._agent_last_laptime = self.agent_driver.predicted_laptime
        if getattr(self, 'reward_relative', False):
            _deltas = [d.total_time - self._tt_before[d.name]
                       for d in self.rival_drivers if not d.is_dnf and d.name in self._tt_before]
            self._field_median_lapdelta = float(np.median(_deltas)) if _deltas else self._agent_last_laptime
            self._agent_lapdelta = self.agent_driver.total_time - self._tt_before[self.agent_driver.name]

        # 6. Update positions and intervals
        self._update_positions_and_intervals()

        # 7. Update driver states (apply mandatory state transitions)
        self._update_driver_states(agent_action, rival_actions, is_yf)

        # 8. Update safety car duration
        self._update_safety_car_duration()

        # Update global state
        self.lap_number += 1
        self.laps_left = max(0, self.n_laps - self.lap_number)

        # Check if race is complete. Race laps are simulated at
        # lap_number = 1..n_laps (n_laps racing laps in total), matching the
        # single-driver environment and the QMIP horizon.
        done = self.lap_number > self.n_laps

        # Calculate reward
        reward = self._calculate_reward(action, done, agent_prev_action)
        if self._no_set_pending:
            reward += self._no_set_pending

        # SC reveal: sortear/leer el SC/VSC de la vuelta siguiente ANTES de
        # construir la observación (no después de la última vuelta).
        if getattr(self, 'sc_reveal', False) and not done:
            self._reveal_next_lap_yf()

        # Update observation cache
        self._update_obs_cache()

        # Include terminal info so evaluation can read it after DummyVecEnv auto-reset
        info = {}
        if done:
            info['final_position'] = self.agent_driver.position
            info['has_changed_compound'] = self.agent_driver.has_changed_compound
            info['total_time'] = self.agent_driver.total_time

        return self._get_obs(), reward, done, False, info

    def _handle_lap_0(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handle lap 0 - agent chooses starting compound."""
        # Map action to compound: action 0 is invalid, 1-3 map to available compounds
        if action == 0 and getattr(self, 'lap0_action0_medium', False):
            # v8 (2026-10-18, opt-in): la accion 0 en la vuelta 0 = compuesto medio
            # (sin sorteo), con la penalizacion lap0_penalty (default la de siempre).
            penalty = getattr(self, 'lap0_penalty', self._penalty_invalid)
            compound = self.available_compounds[len(self.available_compounds) // 2]
        elif action == 0:
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

        # Tyre sets (opt-in): the agent takes the least-worn set of the chosen compound; with none left it is
        # penalised and starts on the middle compound with sets (or any compound with sets)
        if self.tyre_sets:
            ag = self.agent_driver
            if not ag.sets.get(compound):
                penalty += self._no_set_penalty
                with_sets = [c for c in self.available_compounds if ag.sets.get(c)]
                mid = self.available_compounds[len(self.available_compounds) // 2]
                compound = mid if mid in with_sets else (with_sets[0] if with_sets else compound)
            age = tyre_sets.take(ag.sets, compound)
            ag.tyrelife = 0 if age is None else age

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

        # SC reveal: el SC/VSC de la vuelta 1 se conoce antes de decidirla.
        if getattr(self, 'sc_reveal', False):
            self._sc_lap_idx = 0
            self._reveal_next_lap_yf()

        # Update observation cache
        self._update_obs_cache()

        return self._get_obs(), penalty, False, False, {}

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

        # Compuestos de partida de los rivales (opt-in `rival_start_alpha`,
        # 2026-10-18, v8): p ~ Dirichlet(alpha, ..., alpha) una vez por
        # episodio y cada rival parte con p (Dirichlet-multinomial simetrica,
        # alpha ajustado a la vuelta 1 de 2021-23, ver
        # scripts/fit_rival_start_alpha_20261018.py). Misma marginal (1/3 por
        # compuesto) que el sorteo uniforme, pero rivales concentrados como en
        # las carreras reales. p se sortea solo si algun rival no tiene
        # compuesto real (Fig. 7 no cambia). Default (None) = sorteo uniforme.
        alpha = getattr(self, 'rival_start_alpha', None)
        p_start = None

        # Create rival drivers
        for i, name in enumerate(driver_names):
            if name == agent_name:
                continue

            pos = initial_positions[driver_names.index(name)]
            # Evaluation mode: use the real 2024 starting compound when
            # provided (set via set_evaluation_mode), otherwise sample.
            compound = self.eval_initial_compounds.get(name)
            if compound is None or compound not in self.available_compounds:
                if alpha:
                    if p_start is None:
                        p_start = self.np_random.dirichlet([alpha] * len(self.available_compounds))
                    compound = self.available_compounds[
                        int(self.np_random.choice(len(self.available_compounds), p=p_start))]
                else:
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

        # Tyre sets (opt-in): real 2024 sets of every driver; rivals take their starting set now (the agent at
        # lap 0). Rivals with their real 2024 strategy (evaluation) take the real age of each stint instead.
        if self.tyre_sets:
            real = self.eval and bool(self.rival_actions_dict)
            self._real_stint_ages = tyre_sets.race_stint_ages(self.gp) if real else {}
            for d in self.drivers:
                d.sets = tyre_sets.sets_for(self.gp, d.name, self.available_compounds)
            for d in self.rival_drivers:
                if real and d.name in self.rival_actions_dict:
                    d.tyrelife = self._real_stint_ages.get(d.name, {}).get(0, 0)
                else:
                    age = tyre_sets.take(d.sets, d.compound)
                    d.tyrelife = 0 if age is None else age

        # Sort by position
        self.drivers.sort(key=lambda d: d.position)

    def _set_lap_state_age(self, driver: Driver, age: int):
        """Tyre sets (opt-in): tyre life of the lap state = age of the fitted set (the lap-time model reads the
        semantic 'TyreLife' key; the TyreLife x compound column is kept consistent)."""
        st = driver.lap_state
        st['TyreLife'] = float(age)
        col = self._tyre_life_col_map.get(driver.compound)
        if col:
            st[col] = float(age)

    def _initialize_all_lap_states(self):
        """Initialize lap states for all drivers."""
        # Agent
        self.agent_driver.lap_state = initial_lap_v3(
            self.agent_driver.compound,
            self.agent_driver.name,
            self.fast_predictor_clear
        )
        self._initialize_first_lap_variables(self.agent_driver)

        # Compute baseline laptime for centered 'time' reward.
        # The baseline MUST be independent of the agent's action: it is
        # computed with a fixed reference compound (the middle available
        # compound), never with the compound the agent chose at lap 0.
        # Otherwise starting on a slower compound would inflate every
        # per-lap reward (action-dependent shaping).
        if self.reward_type == 'time':
            ref_compound = sorted(self.available_compounds)[len(self.available_compounds) // 2]
            ref_state = initial_lap_v3(
                ref_compound,
                self.agent_driver.name,
                self.fast_predictor_clear
            )
            if 'FirstLap_pos' in ref_state:
                ref_state['FirstLap_pos'] = float(self.agent_driver.initial_pos)
            self._baseline_laptime = self.fast_predictor_clear.predict_single(ref_state)

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

        # Tyre sets (opt-in): the first lap is driven on the starting set, with its age
        if self.tyre_sets:
            for d in self._ordered_drivers:
                d.stint_laps = 0
                self._set_lap_state_age(d, d.tyrelife)

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
        # Reset the first-YF-lap flag; it is set by _activate_vsc/_activate_sc
        # (sampled mode) or recomputed from the real lap lists (eval mode).
        self._is_first_yf_lap = False

        # yf_enabled=False must disable YF events ENTIRELY. Previously only
        # the return value (lap-time effect) was gated while the sampled/real
        # counters were still set, so the agent saw phantom SC/VSC
        # observations and rivals still reacted via XSC/XVSC to events that
        # never physically happened.
        if not self.yf_enabled:
            self.vsc = 0
            self.sc = 0
            self.laps_left_vsc = 0
            self.laps_left_sc = 0
            return False

        # Evaluation mode: SC/VSC activity is derived DIRECTLY from the real
        # lap lists every lap, so events last exactly as long as they did in
        # reality (no fixed-duration countdown). laps_vsc/laps_sc list ALL
        # active laps. The remaining-laps counters are recomputed from the
        # lists so the observation stays meaningful.
        if self.eval:
            prev_lap = self.lap_number - 1
            in_vsc = self.lap_number in self.laps_vsc
            in_sc = (not in_vsc) and self.lap_number in self.laps_sc

            self._is_first_yf_lap = (
                (in_vsc and prev_lap not in self.laps_vsc) or
                (in_sc and prev_lap not in self.laps_sc)
            )

            self.vsc = 1 if in_vsc else 0
            self.sc = 1 if in_sc else 0

            # Remaining consecutive event laps STRICTLY AFTER the current one.
            #
            # Training-path alignment (audited off-by-one fix): in sampled
            # mode _update_safety_car_duration() decrements the counter at the
            # END of the step, BEFORE the observation is built, so during a
            # 4-lap SC the agent observes {3, 2, 1, 0} — i.e. the counter in
            # the observation means "YF laps still to be simulated in FUTURE
            # steps" (0 while the LAST event lap is being run). Counting from
            # self.lap_number + 1 here reproduces exactly that semantics for
            # replayed real events; the previous code counted from
            # self.lap_number (including the current lap) and produced
            # {4, 3, 2, 1}, a +1 shift versus the distribution the policy was
            # trained on. Only the observation counter changes: physical YF
            # activity (self.sc/self.vsc flags, _is_first_yf_lap, YF lap
            # times, TyreLife freeze, rival XSC/XVSC, deterministic
            # overtakes) is driven by in_vsc/in_sc above and is unaffected.
            if in_vsc:
                k = 0
                while (self.lap_number + 1 + k) in self.laps_vsc:
                    k += 1
                self.laps_left_vsc = k
            else:
                self.laps_left_vsc = 0
            if in_sc:
                k = 0
                while (self.lap_number + 1 + k) in self.laps_sc:
                    k += 1
                self.laps_left_sc = k
            else:
                self.laps_left_sc = 0
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
        self._is_first_yf_lap = True

    def _activate_sc(self):
        """Activate Safety Car."""
        self.sc = 1
        self.laps_left_sc = SC_DURATION
        self._is_first_yf_lap = True

    def _reveal_next_lap_yf(self):
        """
        SC reveal (opt-in `sc_reveal`, 2026-10-07): determina el SC/VSC de la
        vuelta self.lap_number (la que el agente está por decidir) al FINAL del
        paso anterior, para que la observación lo muestre.

        Contador observado (incluye la vuelta por decidir):
          - modo estocástico: SC 4,3,2,1 / VSC 2,1 (sale solo del countdown);
          - modo eval (eventos reales): la 1a vuelta del evento muestra siempre
            SC_DURATION / VSC_DURATION (lo único que el agente vio en
            entrenamiento en esa vuelta); las siguientes, las vueltas que
            quedan contando la actual, acotadas a SC_DURATION-1 /
            VSC_DURATION-1. Así el máximo (4 / 2) significa siempre "1a vuelta
            del evento" (un SC real de 5 vueltas no muestra 4 dos veces, uno de
            3 vueltas no oculta su 1a vuelta).
        """
        self._revealed_is_yf = self._update_sc_vsc()
        if self.sc > 0:
            self._sc_lap_idx = getattr(self, '_sc_lap_idx', 0) + 1
        else:
            self._sc_lap_idx = 0
        if self.eval:
            first = bool(self._is_first_yf_lap)
            if self.sc > 0:
                self.laps_left_sc = (SC_DURATION if first
                                     else min(self.laps_left_sc + 1, SC_DURATION - 1))
            if self.vsc > 0:
                self.laps_left_vsc = (VSC_DURATION if first
                                      else min(self.laps_left_vsc + 1, VSC_DURATION - 1))

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
        if is_yf and not self._is_first_yf_lap:
            return np.zeros(num_rivals, dtype=np.int64)

        # No pitting in last 7 laps during normal racing
        if not is_yf and laps_left <= 7:
            return np.zeros(num_rivals, dtype=np.int64)

        # === VECTORIZED PIT STOP DECISION ===
        rival_actions = np.zeros(num_rivals, dtype=np.int64)
        is_sc = bool(self.sc > 0)
        is_vsc = bool(self.vsc > 0)

        # Build mask for eligible drivers (stint length constraint, DNF)
        eligible_mask = np.ones(num_rivals, dtype=bool)
        for i, driver in enumerate(self.rival_drivers):
            if driver.is_dnf:
                eligible_mask[i] = False
            elif not is_yf and driver.stint_laps < MIN_STINT_LENGTH:
                eligible_mask[i] = False

        # If no eligible drivers, return zeros
        if not np.any(eligible_mask):
            return rival_actions

        # Get pit probabilities for all eligible drivers in batch
        if self.rival_pitstop_logit is not None:
            if self._stop_betas is not None:
                pit_probs = self._get_rival_pit_probs_fast(is_sc, is_vsc)
            else:
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

    def _calibrate_rival_fast_path(self):
        """
        Extract closed-form coefficients from the fitted rival logits so the
        per-lap decisions can be computed in pure numpy (fast path).

        LogitStop:  PitIn ~ SC + VSC + used_two_compounds + EarlyLaps +
                    LateLaps + TyreLife:C(Compound_Detail) + Position + GP
        The GP term is constant for this environment and is folded into the
        intercept. Both extractions are validated against the statsmodels
        predict(); any mismatch > 1e-6 disables the fast path.
        """
        self._stop_betas = None
        self._choice_betas = None

        # ---- LogitStop coefficients -------------------------------------
        try:
            p = self.rival_pitstop_logit.params
            gp_term = 0.0
            for name in p.index:
                if name.startswith('GP[T.') and self.gp in name:
                    gp_term = float(p[name])
                    break
            stop = {
                'const': float(p.get('Intercept', 0.0)) + gp_term,
                'SC': float(p.get('SC', 0.0)),
                'VSC': float(p.get('VSC', 0.0)),
                'used': float(p.get('used_two_compounds', 0.0)),
                'early': float(p.get('EarlyLaps', 0.0)),
                'late': float(p.get('LateLaps', 0.0)),
                'pos': float(p.get('Position', 0.0)),
                'tl': np.array([
                    float(p.get(f'TyreLife:C(Compound_Detail)[C{c}]', 0.0))
                    for c in range(6)
                ]),
            }
            # Validate against the statsmodels path on a probe row
            probe = pd.DataFrame([{
                'SC': 0.0, 'VSC': 0.0, 'used_two_compounds': 1.0,
                'EarlyLaps': 0.0, 'LateLaps': 0.0, 'TyreLife': 14.0,
                'Compound_Detail': f'C{self.available_compounds[0]}',
                'Position': 7.0, 'GP': self.gp,
            }])
            p_slow = float(self.rival_pitstop_logit.predict(probe)[0])
            z = (stop['const'] + stop['used'] + 14.0 * stop['tl'][self.available_compounds[0]]
                 + 7.0 * stop['pos'])
            p_fast = 1.0 / (1.0 + np.exp(-z))
            if abs(p_slow - p_fast) < 1e-6:
                self._stop_betas = stop
            elif self.verbose:
                print(f"fast_rivals: LogitStop mismatch ({p_slow} vs {p_fast}), slow path kept")
        except Exception:
            self._stop_betas = None

        # ---- CondLogitCompound coefficients ------------------------------
        try:
            p = self.rival_compound_logit.params
            gp_term = 0.0
            for name in p.index:
                if name.startswith('GP[T.') and self.gp in name:
                    gp_term = float(p[name])
                    break
            alts = {}
            for comp in self.gp_compounds:
                base = p.get(f'C(alt)[{comp}]', None)
                if base is None:
                    raise KeyError(comp)
                ll = p.get(f'C(alt)[{comp}]:LapsLeft',
                           p.get(f'LapsLeft:C(alt)[{comp}]', 0.0))
                alts[comp] = (float(base) + gp_term, float(ll))
            choice = {'alts': alts,
                      'not_change': float(p.get('not_change_compound', 0.0))}
            # Validate on a probe alternative set
            rows = [{'alt': c, 'LapsLeft': 20.0,
                     'not_change_compound': 1.0 if i == 0 else 0.0,
                     'GP': self.gp} for i, c in enumerate(self.gp_compounds)]
            p_slow = np.asarray(self.rival_compound_logit.predict(pd.DataFrame(rows)), dtype=float)
            z = np.array([
                alts[c][0] + 20.0 * alts[c][1] + (choice['not_change'] if i == 0 else 0.0)
                for i, c in enumerate(self.gp_compounds)
            ])
            p_fast = 1.0 / (1.0 + np.exp(-z))
            if np.max(np.abs(p_slow - p_fast)) < 1e-6:
                self._choice_betas = choice
            elif self.verbose:
                print(f"fast_rivals: CondLogit mismatch ({p_slow} vs {p_fast}), slow path kept")
        except Exception:
            self._choice_betas = None

    def _get_rival_pit_probs_fast(self, is_sc: bool, is_vsc: bool) -> np.ndarray:
        """Pure-numpy LogitStop evaluation for all rivals (fast path)."""
        b = self._stop_betas
        n = len(self.rival_drivers)
        z = np.full(n, b['const']
                    + (b['SC'] if is_sc else 0.0)
                    + (b['VSC'] if is_vsc else 0.0)
                    + (b['early'] if self.lap_number <= 10 else 0.0)
                    + (b['late'] if self.lap_number >= (self.n_laps - 10) else 0.0))
        for i, d in enumerate(self.rival_drivers):
            z[i] += (b['used'] if d.has_changed_compound else 0.0) \
                    + d.tyrelife * b['tl'][d.compound] \
                    + d.position * b['pos']
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def _predict_compound_choice_fast(self, driver: Driver, laps_left: int) -> int:
        """Pure-numpy CondLogitCompound evaluation + sampling (fast path)."""
        c = self._choice_betas
        z = np.empty(self._n_compounds)
        for i, comp in enumerate(self.gp_compounds):
            base, ll = c['alts'][comp]
            z[i] = base + laps_left * ll
            if comp == f'C{driver.compound}' and not driver.has_changed_compound:
                z[i] += c['not_change']
        probs = 1.0 / (1.0 + np.exp(-z))
        probs = probs / probs.sum()
        if self.tyre_sets:
            probs = self._mask_no_set(driver, probs)
        chosen = self.np_random.choice(self.gp_compounds, p=probs)
        return int(chosen[1:])

    def _mask_no_set(self, driver: Driver, probs: np.ndarray) -> np.ndarray:
        """Tyre sets (opt-in): compound-choice probabilities restricted to the compounds with sets left
        (renormalised; unchanged if no compound has sets)."""
        ok = np.array([1.0 if self._has_set(driver, c) else 0.0 for c in self.available_compounds])
        q = probs * ok
        return q / q.sum() if q.sum() > 0 else probs

    def _has_set(self, driver: Driver, compound: int) -> bool:
        """Tyre sets (opt-in): a set of the compound can be fitted at a stop (with reuse, the mounted compound
        always can: its own set comes back)."""
        return tyre_sets.has_set(driver.sets, compound, (driver.compound,) if self._reuse else None)

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

        # Fast path: closed-form coefficients extracted at init
        if self._choice_betas is not None:
            return self._predict_compound_choice_fast(driver, laps_left)

        # 2026-10-21: NumPy path for the softmax conditional logit (CondLogitCompoundView, v7+). Same float64
        # operations as its predict() on the DataFrame below and the same RNG call, so the draws are identical;
        # it only avoids building a DataFrame and looking coefficients up by name on every stop.
        model = self.rival_compound_logit
        if getattr(model, 'choice_rule', None) == 'softmax':
            cache = getattr(self, '_softmax_choice_cache', None)
            if cache is None or cache[0] is not model:
                p = model.params
                base = np.array([p.get(f'C(alt)[{a}]', np.nan) for a in self.gp_compounds], dtype=float)
                ll = np.array([p.get(f'C(alt)[{a}]:LapsLeft', 0.0) for a in self.gp_compounds], dtype=float)
                cache = (model, base, ll, p.get('not_change_compound', 0.0))
                self._softmax_choice_cache = cache
            _, base, ll, nc_coef = cache
            cur = f'C{driver.compound}'
            nc = np.array([1.0 if (c == cur and not driver.has_changed_compound) else 0.0
                           for c in self.gp_compounds], dtype=float)
            try:
                v = base + ll * np.full(len(nc), float(laps_left)) + nc_coef * nc
                e = np.exp(v - np.nanmax(v))
                probs = e / np.nansum(e)
                probs = probs / probs.sum()
                if self.tyre_sets:
                    probs = self._mask_no_set(driver, probs)
                chosen = self.np_random.choice(self.gp_compounds, p=probs)
                return int(chosen[1:])
            except Exception:
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
            if self.tyre_sets:
                probs = self._mask_no_set(driver, probs)

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
                    if self.tyre_sets:
                        available_options = [c for c in available_options
                                             if self._has_set(self.rival_drivers[i], c)] or available_options
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
            if rival.is_dnf:
                continue
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
        # First racing lap: predict with the initial lap state exactly as
        # built by initial_lap_v3 (LapNumber=1, FirstLap_pos=grid position,
        # TyreLife=0). Only the pit-in flag can differ from the initial state.
        # Without this branch the first simulated lap would be predicted with
        # LapNumber=2 and FirstLap_pos active — a combination that never
        # occurs in the training data — and lap 1 would never be simulated.
        if self.lap_number == 1:
            actions = np.concatenate([[agent_action], rival_actions])
            for driver, act in zip(self._ordered_drivers, actions):
                if 'PitIn' in driver.lap_state:
                    driver.lap_state['PitIn'] = 1.0 if act != 0 else 0.0
            return

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
        if self.tyre_sets and self.agent_driver.prev_action != 0:
            self._set_lap_state_age(self.agent_driver, self.agent_driver.next_set_age)

        # Rivals - pass O(1) lookup maps for maximum performance
        for i, driver in enumerate(self.rival_drivers):
            if driver.is_dnf:
                continue
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
            if self.tyre_sets and driver.prev_action != 0:
                self._set_lap_state_age(driver, driver.next_set_age)
        self._charge_last_lap_pitout()

    def _charge_last_lap_pitout(self):
        """Cierre del loophole del pit en la ÚLTIMA vuelta (opt-in
        `last_lap_pit_full_cost`, 2026-09-03). En el modelo de tiempos el pit
        se paga en dos vueltas (PitIn en la de entrada, PitOut en la de
        salida); en la vuelta final no existe la de salida, así que un pit
        allí costaba solo PitIn (~5 s en vez de ~20 s) y satisfacía la regla
        de dos compuestos casi gratis. Medido en el árbol dsq1000: PPO Saudi
        90 %, TRPO Miami 95 % de los episodios paraban en la última vuelta
        (65 % PPO Saudi ya en el árbol yf2024). Con el flag, todo piloto con
        PitIn en la vuelta final carga también PitOut en esa misma vuelta
        (delta completo de pit-lane, como en la realidad)."""
        if not getattr(self, 'last_lap_pit_full_cost', False):
            return
        if self.lap_number < self.n_laps:
            return
        for d in self._ordered_drivers:
            if d.lap_state.get('PitIn', 0.0) and 'PitOut' in d.lap_state:
                d.lap_state['PitOut'] = 1.0

    @property
    def _active_drivers(self) -> List[Driver]:
        """Drivers still running (excludes retired rivals in evaluation mode)."""
        return [d for d in self.drivers if not d.is_dnf]

    def _calculate_intervals(self) -> List[List[float]]:
        """Calculate intervals between running drivers (indexed by position - 1)."""
        sorted_drivers = sorted(self._active_drivers, key=lambda d: d.position)
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

        # SC bunching (spec del autor 2026-08-28): la PRIMERA vuelta de SC usa
        # el coeficiente SC plano, NO el de SC_firstlap. Solo se suprime la
        # FEATURE del modelo de tiempos; el flag _is_first_yf_lap queda intacto
        # porque ademas gobierna la ventana de pit de los rivales.
        _first_yf_feat = self._is_first_yf_lap
        if getattr(self, 'sc_bunching', False) and self.sc > 0:
            _first_yf_feat = False

        for i, driver in enumerate(all_drivers):
            if driver.is_dnf:
                # Retired drivers no longer set lap times
                continue
            if is_yf and self.fast_predictor_yf is not None:
                # Yellow flag model
                features = build_yf_features(
                    driver.lap_state,
                    is_sc=(self.sc > 0),
                    is_vsc=(self.vsc > 0),
                    is_first_yf_lap=_first_yf_feat,
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

        # NOTE: the agent laptime used by the reward is set in step() AFTER
        # _resolve_overtakes(), so that traffic adjustments are included.
        return laptimes

    def _sample_following_gap(self) -> float:
        """Gap at which a car that could NOT pass settles behind the car ahead.

        Legacy behaviour (following_gap_scale = 0) returns the constant MIN_GAP,
        which welds trains at exactly 0.4 s. With a positive scale the gap is
        MIN_GAP + Exponential(scale), resampled per pair per lap, so the field
        spreads the way the real Interval_front data does. Uses the env RNG, so
        runs stay reproducible and paired across counterfactual arms."""
        scale = getattr(self, 'following_gap_scale', 0.0)
        if not scale:
            return MIN_GAP
        return MIN_GAP + float(self.np_random.exponential(scale))

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
        # Sort RUNNING drivers by position at start of lap (retired rivals
        # neither set lap times nor take part in overtakes)
        sorted_drivers = sorted(self._active_drivers, key=lambda d: d.position)
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

        # Opt-in pit-lane pass (see PIT_PASS_DETERMINISTIC_DEFAULT): drivers
        # whose current lap carries the PitIn/PitOut penalty are outside the
        # overtake logit's support, so their battles resolve deterministically.
        pit_pass_det = getattr(self, 'pit_pass_deterministic',
                               PIT_PASS_DETERMINISTIC_DEFAULT)
        pit_affected = ({d.name for d in sorted_drivers
                         if d.lap_state.get('PitIn', 0.0)
                         or d.lap_state.get('PitOut', 0.0)}
                        if pit_pass_det else frozenset())

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
                elif gap_k_to_k1 > OVERTAKE_MAX_GAP:
                    # Outside the logit's training support (pairs > 1.0s
                    # apart at the start of the lap): no on-track battle.
                    # The attacker simply ends up ahead if it is faster by
                    # more than the gap (e.g. the defender pitted).
                    overtake = pred_gap < 0
                elif (driver_k.name in pit_affected
                      or driver_k_1.name in pit_affected):
                    # Opt-in pit-lane pass: one of the pair is on an in-/out-lap
                    # (pred_gap carries the ~10-24s PitIn/PitOut penalty), also
                    # outside the logit's support even though the start-of-lap
                    # gap is within 1.0s. Resolve deterministically so a car in
                    # the pit lane cannot "defend" and weld the train behind it.
                    # (pit_affected is empty unless pit_pass_deterministic.)
                    overtake = pred_gap < 0
                else:
                    # Clear track battle (gap <= 1.0s): stochastic overtake
                    # using the fitted logit τ from the batch lookup.
                    pair_key = (driver_k.name, driver_k_1.name)
                    if pair_key in tau_lookup:
                        tau = tau_lookup[pair_key]
                    elif self.logit_overtake is not None:
                        # Pair created by a swap earlier in this lap: use the
                        # logit calibrated at init (same intercept/slope).
                        tau = self._quick_tau_estimate(pred_gap)
                    else:
                        tau = 0.7 if pred_gap < -0.5 else 0.3

                    # CRITICAL: Overtake ← Uniform(0,1) < τ (STOCHASTIC)
                    overtake = random_thresholds[random_idx] < tau
                    random_idx += 1

                if overtake:
                    # === OVERTAKE OCCURS ===
                    _mg = self._sample_following_gap
                    # ŷ_d(k) ← max{ŷ_d(k), ŷ_d(k-2) − X_Δ2^(k,k-2) + t_min_gap}
                    # When k < 2 (overtaking into P1), no car ahead → no constraint
                    if k >= 2:
                        driver_k_2 = sorted_drivers[k-2]
                        gap_k_to_k2 = original_total_times[driver_k.name] - original_total_times[driver_k_2.name]
                        pred_laptimes[driver_k.name] = max(
                            pred_laptimes[driver_k.name],
                            pred_laptimes[driver_k_2.name] - gap_k_to_k2 + _mg()
                        )

                    # ŷ_d(k-1) ← max{ŷ_d(k-1), ŷ_d(k) + X_Δ2^(k,k-1) + t_min_gap}
                    pred_laptimes[driver_k_1.name] = max(
                        pred_laptimes[driver_k_1.name],
                        pred_laptimes[driver_k.name] + gap_k_to_k1 + _mg()
                    )

                    # Swap d^(k) and d^(k-1) in sorted array
                    sorted_drivers[k], sorted_drivers[k-1] = sorted_drivers[k-1], sorted_drivers[k]
                    k -= 1
                else:
                    # === NO OVERTAKE ===
                    pred_laptimes[driver_k.name] = max(
                        pred_laptimes[driver_k.name],
                        pred_laptimes[driver_k_1.name] - gap_k_to_k1
                        + self._sample_following_gap()
                    )
                    break

        # Apply adjusted predicted lap times to total times (running drivers)
        for driver in sorted_drivers:
            original_time_before_lap = original_total_times[driver.name]
            driver.total_time = original_time_before_lap + pred_laptimes[driver.name]
            driver.predicted_laptime = pred_laptimes[driver.name]

    def _compact_sc_gaps(self, factor):
        """Compactación parcial bajo SC (vuelta 2 del período): los gaps al
        líder se escalan por `factor` (<1) preservando el orden. Mismas
        salvaguardas que la fila completa: sin super-velocidad para los muy
        rezagados (media vuelta SC de avance máximo) y monotonía estricta."""
        delta = float(getattr(self, 'sc_bunch_delta', 1.5))
        drivers = sorted(self._active_drivers, key=lambda d: d.position)
        leader = drivers[0]
        d_min = min(delta, MIN_GAP)
        max_catch = 0.5 * leader.predicted_laptime
        prev = leader.total_time
        for d in drivers[1:]:
            gap = d.total_time - leader.total_time
            new_total = max(leader.total_time + factor * gap,
                            d.total_time - max_catch,
                            prev + d_min)
            d.predicted_laptime = d.predicted_laptime - (d.total_time - new_total)
            d.total_time = new_total
            prev = new_total

    def _apply_sc_bunching(self):
        """Vueltas 3+ de un período SC (opt-in `sc_bunching`): el pelotón se
        APILA detrás del líder y los gaps acumulados se borran.

        El líder avanza con su laptime SC (ya calculado en el paso 4); el
        coche en la posición p queda con total = t_lider + (p-1)·delta,
        preservando el orden de posiciones vigente (sin duelos: bajo SC no se
        adelanta). Dos salvaguardas:
        - un coche muy rezagado (p.ej. doblado) ALCANZA la fila de forma
          progresiva: su vuelta no puede ser más corta que la mitad del
          laptime SC del líder (evita laptimes negativos y, en entrenamiento,
          reward-hacking por "teletransporte");
        - monotonía estricta de totales para que el ranking por tiempo
          preserve el orden aunque el clamp anterior dispare.
        """
        delta = float(getattr(self, 'sc_bunch_delta', 1.5))
        drivers = sorted(self._active_drivers, key=lambda d: d.position)
        leader = drivers[0]
        lt = leader.predicted_laptime
        min_lap = 0.5 * lt
        d_min = min(delta, MIN_GAP)

        leader.total_time = leader.total_time + lt
        prev_total = leader.total_time
        for i, d in enumerate(drivers[1:], start=1):
            old = d.total_time
            new_total = max(leader.total_time + i * delta,   # su lugar en la fila
                            old + min_lap,                    # sin super-velocidad
                            prev_total + d_min)               # monotonía
            d.predicted_laptime = new_total - old
            d.total_time = new_total
            prev_total = new_total

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

        # Fast path: the overtake model is a 2-parameter logit, so the
        # calibrated (intercept, slope) reproduce it EXACTLY — vectorized
        # sigmoid instead of DataFrame + statsmodels predict.
        if self.fast_rivals:
            z = np.clip(self._ot_intercept + self._ot_slope * np.asarray(pred_gaps),
                        -500, 500)
            taus = 1.0 / (1.0 + np.exp(-z))
            return dict(zip(pairs, taus))

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
        """
        τ estimate when the batch lookup misses (pairs created by swaps).

        Uses the intercept/slope of THIS GP's fitted overtake logit
        (extracted once at init via _calibrate_overtake_logit), so swapped
        pairs face exactly the same model as the batch-predicted pairs.
        """
        z = np.clip(self._ot_intercept + self._ot_slope * pred_gap, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _calibrate_overtake_logit(self):
        """
        Extract (intercept, slope) in logit space from the fitted overtake
        model by probing it at delta_total = 0 and 1. Falls back to a mild
        generic curve if the model cannot be probed.
        """
        self._ot_intercept = 0.0
        self._ot_slope = -1.5  # generic fallback: decreasing in pred_gap
        if self.logit_overtake is None:
            return
        try:
            probe = pd.DataFrame({
                'const': [1.0, 1.0],
                'delta_total': [0.0, 1.0],
            })
            p = np.clip(np.asarray(self.logit_overtake.predict(probe), dtype=float),
                        1e-9, 1 - 1e-9)
            logit = np.log(p / (1 - p))
            self._ot_intercept = float(logit[0])
            self._ot_slope = float(logit[1] - logit[0])
        except Exception:
            pass

    def _update_positions_and_intervals(self):
        """
        Update positions based on total race time.

        State transition: S_Position[d] <- rank(cumulative_time)

        Retired drivers (evaluation mode) are classified behind all running
        drivers, in retirement order.
        """
        active = [d for d in self.drivers if not d.is_dnf]
        retired = [d for d in self.drivers if d.is_dnf]
        active.sort(key=lambda d: d.total_time)
        self.drivers = active + retired
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
            if driver.is_dnf:
                continue
            old_compound = driver.compound

            if action == 0:
                # No pit stop - increment tyre life unless under yellow flag
                if not is_yf:
                    driver.tyrelife += 1
                    driver.stint_laps += 1
                # S_Tire[d] stays the same
            else:
                # Pit stop - reset tyre life and update compound
                # Reuse (opt-in): the set taken off goes back with its age (this lap counted unless under SC/VSC)
                if self.tyre_sets and self._reuse and old_compound != 0:
                    tyre_sets.give_back(driver.sets, old_compound, driver.tyrelife + (0 if is_yf else 1))
                driver.tyrelife = 0
                driver.stint_laps = 0
                driver.compound = action
                # Tyre sets (opt-in): the new stint starts at the age of the fitted set (real 2024 age for rivals
                # with their real strategy, else the least-worn set of the compound)
                if self.tyre_sets:
                    real = self._real_stint_ages.get(driver.name) if driver is not self.agent_driver else None
                    if real is not None and driver.name in self.rival_actions_dict:
                        age = real.get(int(self.lap_number), 0)
                    else:
                        age = tyre_sets.take(driver.sets, action)
                    driver.tyrelife = driver.next_set_age = 0 if age is None else int(age)

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
        # Evaluation mode: activity and remaining laps are recomputed from the
        # real lap lists in _update_sc_vsc(); the fixed-duration countdown
        # must not deactivate (or extend) real events.
        if self.eval:
            return
        if self.vsc > 0:
            self.laps_left_vsc = max(0, self.laps_left_vsc - 1)
            if self.laps_left_vsc == 0:
                self.vsc = 0
        if self.sc > 0:
            self.laps_left_sc = max(0, self.laps_left_sc - 1)
            if self.laps_left_sc == 0:
                self.sc = 0

    def _calculate_reward(self, action: int, done: bool, agent_prev_action: int = 0) -> float:
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
            if getattr(self, 'reward_relative', False):
                # mix-v2: laptime del agente RELATIVO a la mediana del pelotón en la
                # misma vuelta. Cancela los shocks comunes (SC/VSC suman +30 s a
                # todos) que hoy dominan la varianza del retorno (≈-5000 ± cientos)
                # y ahogan la señal de decisión (una 2a parada vale 30-130).
                reward += -(self._agent_lapdelta - self._field_median_lapdelta)
            else:
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

        # Consecutive pit penalty (all reward types).
        # agent_prev_action is the action of the PREVIOUS lap, snapshotted in
        # step() before _update_driver_states() overwrites driver.prev_action.
        if action > 0 and agent_prev_action > 0:
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

        # Time differences to agent (rivals only), clipped to the declared
        # observation bounds. Retired rivals are pinned to the "far behind"
        # bound so their frozen cumulative time does not read as a huge lead.
        for i, d in enumerate(self.rival_drivers):
            if d.is_dnf:
                self._obs_cache["time_diff_to_agent"][i] = 500.0
            else:
                self._obs_cache["time_diff_to_agent"][i] = float(
                    np.clip(d.total_time - agent_time, -500.0, 500.0)
                )

        # Scalars. SC/VSC counters are capped at the declared space bounds
        # (real events replayed in evaluation mode can outlast the nominal
        # SC_DURATION/VSC_DURATION).
        self._obs_cache["lap_number"] = self.lap_number
        self._obs_cache["laps_left"] = self.laps_left
        self._obs_cache["prev_action"] = self.agent_driver.prev_action if self.agent_driver else 0
        self._obs_cache["vsc"] = min(self.laps_left_vsc, VSC_DURATION) if self.vsc > 0 else 0
        self._obs_cache["sc"] = min(self.laps_left_sc, SC_DURATION) if self.sc > 0 else 0
        if self.tyre_sets and self.agent_driver is not None and self.agent_driver.sets is not None:
            ag = self.agent_driver
            self._obs_cache["tyre_sets"][:] = (
                tyre_sets.observation_reuse(ag.sets, self.available_compounds, (ag.compound, ag.tyrelife + 1))
                if self._reuse else tyre_sets.observation(ag.sets, self.available_compounds))

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
        rival_actions: Dict = None,
        rival_initial_compounds: Dict = None,
    ):
        """
        Configure environment for evaluation with predetermined events.

        Args:
            rival_initial_compounds: {driver_name: compound_int} real starting
                compounds. Applied in reset(), BEFORE the first observation,
                so the agent's lap-0 compound decision is conditioned on the
                real grid instead of randomly sampled rival compounds.
        """
        self.eval = eval_mode
        self.laps_vsc = laps_vsc or []
        self.laps_sc = laps_sc or []
        if initial_positions:
            self.initial_positions = initial_positions
        self.rival_actions_dict = rival_actions or {}
        self.eval_initial_compounds = rival_initial_compounds or {}

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
        """Returns current race classification (retired drivers last)."""
        sorted_drivers = sorted(self.drivers, key=lambda d: (d.is_dnf, d.total_time))
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
    fast_rivals: bool = False,
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

    env = F1EnvAllDrivers(
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
        fast_rivals=fast_rivals,
    )
    # Opt-in por VARIABLES DE ENTORNO (2026-08-18): aplica los fixes de física
    # a los scripts legacy (evaluate_qmip_vs_drl, evaluate_all_gps_with_qmip,
    # rank_showcase_laps, ...) sin tocar sus firmas. Default (sin variables) =
    # comportamiento legacy exacto. Los scripts nuevos que asignan estos
    # atributos EXPLÍCITAMENTE lo hacen después de esta factory, así que sus
    # flags siempre prevalecen sobre las variables de entorno.
    _fgs = os.environ.get('F1_FOLLOWING_GAP_SCALE')
    if _fgs:
        env.following_gap_scale = float(_fgs)
    if os.environ.get('F1_PIT_DET_PASS'):
        env.pit_pass_deterministic = True
    # SC bunching (2026-08-28): aglomeración del pelotón en vueltas 3+ de SC
    # + pit solo en la 1a vuelta de SC + coef SC plano (sin SC_firstlap).
    if os.environ.get('F1_SC_BUNCHING'):
        env.sc_bunching = True
    _sbd = os.environ.get('F1_SC_BUNCH_DELTA')
    if _sbd:
        env.sc_bunch_delta = float(_sbd)
    if os.environ.get('F1_RULE_MASK'):
        env.rule_mask = True
    # Investigación Miami 2026-09-02: escala de la penalización de DSQ (el
    # -5000 del mix domina el gradiente; la consecuencia natural = P20).
    _dsq = os.environ.get('F1_DSQ_PENALTY')
    if _dsq:
        env._penalty_disqualification = float(_dsq)
    if os.environ.get('F1_LASTLAP_PITOUT'):
        env.last_lap_pit_full_cost = True
    if os.environ.get('F1_REWARD_RELATIVE'):
        env.reward_relative = True
    # SC reveal (2026-10-07): el SC/VSC de la vuelta L se revela antes de que
    # el agente decida la vuelta L (ver _reveal_next_lap_yf).
    if os.environ.get('F1_SC_REVEAL'):
        env.sc_reveal = True
    # Abandonos de rivales (2026-10-14): F1_RIVAL_DNF = ruta del JSON de
    # scripts/rival_dnf_calibration_20261014.py (ver reset).
    _dnf = os.environ.get('F1_RIVAL_DNF')
    if _dnf:
        import json
        with open(_dnf) as _f:
            env.rival_dnf = json.load(_f)
    # Compuestos de partida de rivales Dirichlet-multinomial (2026-10-18, v8):
    # F1_RIVAL_START_ALPHA = alpha simetrico (ver _initialize_race_state).
    _rsa = os.environ.get('F1_RIVAL_START_ALPHA')
    if _rsa:
        env.rival_start_alpha = float(_rsa)
    # Accion 0 en la vuelta 0 (2026-10-18, v8): F1_LAP0_ACTION0_MEDIUM=1 la
    # convierte en el compuesto medio; F1_LAP0_PENALTY = su penalizacion.
    if os.environ.get('F1_LAP0_ACTION0_MEDIUM'):
        env.lap0_action0_medium = True
    _l0p = os.environ.get('F1_LAP0_PENALTY')
    if _l0p:
        env.lap0_penalty = float(_l0p)
    # Largada sorteada (2026-10-25, post-entrenamiento): F1_START_MIX = prob. de
    # que el entorno juegue la vuelta 0 con un compuesto uniforme (ver reset).
    _sm = os.environ.get('F1_START_MIX')
    if _sm:
        env.start_mix = float(_sm)
    return env


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
