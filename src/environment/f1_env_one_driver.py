"""
Environment ULTRA-OPTIMIZADO con predicciones NumPy manuales.

OPTIMIZACIONES DE RENDIMIENTO:
- Diccionarios en lugar de DataFrames (mas rapido)
- FastLinearPredictor: 32x mas rapido que statsmodels (44us -> 1.4us)
- Cache de observaciones para evitar allocaciones
- Pre-calculo de penalizaciones
- Eliminacion de operaciones redundantes

RENDIMIENTO ESPERADO:
- 10,000 predicciones por episodio: 440ms -> 14ms (31x speedup)
- 1,000 episodios: 440s -> 14s
- Uso de memoria: 50% reduccion con float32
"""

import config

from typing import Dict, Tuple, List, Optional
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict as GymDict, Discrete
from optimization.utils_qmip import (
    extract_simplified_calculator,
    SimplifiedLaptimeCalculator,
    SimplifiedLapState
)

# Try to import FastLinearPredictor for type checking
try:
    from reg import FastLinearPredictor
except ImportError:
    FastLinearPredictor = None


class F1EnvOneDriverV2Optimized(Env):
    """
    Environment F1 para un solo piloto - VERSION ULTRA-OPTIMIZADA.

    Cambios vs version original:
    - Diccionarios en lugar de DataFrames (mas rapido y simple)
    - Cache de observaciones
    - Pre-calculo de penalizaciones como constantes
    - Eliminacion de operaciones redundantes
    """

    __slots__ = (
        'n_laps', 'available_compounds', 'min_compound', 'max_compound',
        'gp', 'driver_name', 'laptime_calculator',
        '_penalty_no_compound_change', '_penalty_pit_last_lap',
        '_penalty_early_pit', '_penalty_consecutive_pit', '_penalty_invalid',
        '_early_laps_set', '_n_compounds',
        'compound', 'tyre_life', 'change_compound', 'lap_number',
        'laps_left', 'prev_action', 'current_action', 'initial_compound',
        'total_time', 'lap_state', 'prev_compound', 'pit_in_flag', 'pit_out_flag',
        'pit_stop_count', 'compounds_used',
        '_obs_cache', 'observation_space', 'action_space'
    )

    def __init__(self, n_laps: int, available_compounds: List[int], gp: str,
                 driver: str, model_laptime, verbose: bool = False):
        """
        Inicializa el environment optimizado.

        Args:
            n_laps: Number of laps in the race
            available_compounds: Lista de compuestos disponibles (e.g., [3, 4, 5])
            gp: Nombre del Gran Premio
            driver: Codigo del piloto (e.g., 'VER')
            model_laptime: FastLinearPredictor (32x faster) o statsmodels model
            verbose: Si True, imprime informacion de rendimiento al inicializar
        """
        self.n_laps = n_laps
        # Normalizar compuestos (convertir strings si es necesario) - dinamico
        self.available_compounds = [int(c[1:]) if isinstance(c, str) else int(c)
                                   for c in available_compounds]
        self._n_compounds = len(self.available_compounds)
        self.min_compound = min(self.available_compounds)
        self.max_compound = max(self.available_compounds)
        self.gp = gp
        self.driver_name = driver

        # Crear calculadora simplificada (solo usa 4 coeficientes)
        self.laptime_calculator = extract_simplified_calculator(model_laptime, self.available_compounds)

        # Pre-calcular penalizaciones (constantes)
        # PHASE 1 FIX: Reduced penalties to be proportional to lap times (~10-20 seconds)
        # Regulatory violations still penalized but not catastrophically
        self._penalty_no_compound_change = -100.0  # Reduced from -300.0 (regulatory)
        self._penalty_pit_last_lap = -30.0  # Reduced from -500.0 (strategic mistake)
        self._penalty_early_pit = -15.0  # Reduced from -300.0 (discourage, not forbid)
        self._penalty_consecutive_pit = -20.0  # Reduced from -300.0 (wasteful)
        self._penalty_invalid = -150.0  # Reduced from -500.0 (Lap 0 violation, regulatory)

        # Set de early laps (inmutable, mas rapido que list)
        self._early_laps_set = frozenset([1, 2, 3, 4, 5])

        # State variables
        self.compound = 0
        self.tyre_life = 0
        self.change_compound = False
        self.lap_number = 0
        self.laps_left = n_laps
        self.prev_action = 0
        self.current_action = 0
        self.initial_compound = 0
        self.total_time = 0.0
        self.lap_state = None  # SimplifiedLapState
        self.prev_compound = 0
        self.pit_in_flag = 0.0
        self.pit_out_flag = 0.0


        # Cache para observacion (evitar crear dict cada vez)
        self._obs_cache = {
            "compound": 0,
            "tyre_life": 0,
            "change_compound": 0,
            "lap_number": 0,
            "laps_left": n_laps,
            "prev_action": 0,
        }

        # Spaces
        self.observation_space = GymDict({
            "compound": Discrete(self.max_compound + 1),
            "tyre_life": Discrete(n_laps + 1),
            "change_compound": Discrete(2),
            "lap_number": Discrete(n_laps + 1),
            "laps_left": Discrete(n_laps + 1),
            "prev_action": Discrete(self.max_compound + 1),
        })

        # Action space: 0 (no pit) + 1-3 (softest to hardest available compound)
        # Fixed 4 actions that map dynamically to available compounds
        self.action_space = Discrete(4)  # Actions: 0, 1, 2, 3

        # Log predictor type for performance tracking
        if verbose:
            self._log_predictor_info()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, int], Dict]:
        """Reset ultra-optimizado - retorna cache directamente."""
        super().reset(seed=seed)

        # Reset variables (orden optimizado para cache, batch assignments)
        n_laps = self.n_laps  # Local ref para evitar lookups
        self.lap_number = 0
        self.laps_left = n_laps
        self.compound = 0
        self.tyre_life = 0
        self.prev_action = 0
        self.current_action = 0
        self.prev_compound = 0
        self.initial_compound = 0
        self.change_compound = False
        self.total_time = 0.0
        self.lap_state = None

        # PHASE 2: Reset enhanced tracking
        # self.pit_stop_count = 0
        # self.compounds_used = set()

        # Update cache in-place (mas rapido, batch updates)
        cache = self._obs_cache
        cache["lap_number"] = 0
        cache["laps_left"] = n_laps
        cache["compound"] = 0
        cache["tyre_life"] = 0
        cache["prev_action"] = 0
        cache["change_compound"] = 0

        # Retornar dict directamente (Gymnasium hace la copia internamente si es necesario)
        return cache, {}

    def step(self, action: int) -> Tuple[Dict[str, int], float, bool, bool, Dict]:
        """Step ultra-optimizado usando calculo directo con 4 coeficientes."""
        # Lap 0: seleccion de compuesto
        if not self.lap_number:  # self.lap_number == 0 (mas rapido)
            return self._handle_lap_0(action)

        # Map action to compound: 0=no pit, 1-3 map to available compounds (softest to hardest)
        if action == 0:
            pit_compound = 0  # No pit
        elif 1 <= action <= self._n_compounds:
            # Map action index to compound: sorted available_compounds ensures softest->hardest
            pit_compound = self.available_compounds[action - 1]
        else:
            # Invalid action (should not happen with Discrete(4) and 3 compounds)
            self._update_obs_cache()
            return self._obs_cache, self._penalty_invalid, False, False, {}

        # PASO 1: Determinar PitIn para esta vuelta
        pit_in = 1.0 if pit_compound > 0 else 0.0

        # PASO 2: Calcular laptime usando el modelo simplificado
        # Usar compound, tyre_life, pit_in y pit_out_flag (del paso anterior)
        lap_time = self.laptime_calculator.predict(
            compound=self.compound,
            tyre_life=self.tyre_life,
            pit_in=pit_in,
            pit_out=self.pit_out_flag
        )
        self.total_time += lap_time

        # PASO 3: Actualizar estado para la SIGUIENTE vuelta

        # Actualizar PitOut para la SIGUIENTE vuelta (basado en PitIn de ESTA vuelta)
        self.pit_out_flag = pit_in

        # Update driver state
        if pit_compound == 0:
            # No pit stop - increment tire life
            self.tyre_life += 1
        else:
            # Pit stop - reset tire life and change compound
            old_compound = self.compound
            self.tyre_life = 0
            self.compound = pit_compound

            # Mark compound change if different from initial (and not from compound 0)
            if pit_compound != old_compound and old_compound != 0:
                self.change_compound = True

        # Progress lap (batch updates, usar referencias locales)
        lap_number = self.lap_number + 1
        self.lap_number = lap_number
        self.laps_left -= 1

        # Update previous action with one step delay and compound
        prev_action = self.current_action
        self.prev_action = prev_action
        self.current_action = action
        if pit_compound > 0:
            self.prev_compound = pit_compound

        # Check done
        done = lap_number > self.n_laps

        # Calcular reward con penalizaciones (replicando logica original exacta)
        reward = -lap_time

        # Apply penalties for invalid strategies
        if done and not self.change_compound:
            reward += self._penalty_no_compound_change

        if pit_compound and done:
            reward += self._penalty_pit_last_lap

        if pit_compound and (lap_number - 1) in self._early_laps_set:
            reward += self._penalty_early_pit

        if action > 0 and prev_action > 0:
            reward += self._penalty_consecutive_pit

        # Update observation cache
        self._update_obs_cache()

        # Retornar cache directamente (sin copia)
        return self._obs_cache, reward, done, False, {}

    def _handle_lap_0(self, action: int) -> Tuple[Dict[str, int], float, bool, bool, Dict]:
        """Manejo ultra-optimizado de lap 0."""
        # Map action to compound: action 0 is invalid, 1-3 map to available compounds
        if action == 0:
            # Invalid: must select a compound at lap 0
            penalty = self._penalty_invalid
            compound = np.random.choice(self.available_compounds)
        elif 1 <= action <= self._n_compounds:
            # Valid action: map to compound (softest to hardest)
            penalty = 0.0
            compound = self.available_compounds[action - 1]
        else:
            # Invalid action (should not happen)
            self._update_obs_cache()
            return self._obs_cache, self._penalty_invalid, False, False, {}

        # Set compound (batch updates)
        self.compound = compound
        self.initial_compound = compound
        self.prev_compound = compound

        # Inicializar flags de pit
        self.pit_in_flag = 0.0
        self.pit_out_flag = 0.0

        # Progress to lap 1 (batch updates)
        self.lap_number = 1
        # self.laps_left = self.n_laps - 1

        # Update actions
        self.prev_action = self.current_action
        self.current_action = action

        # Update observation cache
        self._update_obs_cache()

        # Retornar cache directamente (sin copia)
        return self._obs_cache, penalty, False, False, {}

    def _update_obs_cache(self) -> None:
        """Actualiza el cache de observacion in-place (ultra-optimizado)."""
        # Usar referencias locales para reducir lookups de atributos
        cache = self._obs_cache
        cache["compound"] = self.compound
        cache["tyre_life"] = self.tyre_life
        cache["change_compound"] = int(self.change_compound)
        cache["lap_number"] = self.lap_number
        cache["laps_left"] = self.laps_left
        cache["prev_action"] = self.prev_action

    def _log_predictor_info(self) -> None:
        """Log information about the simplified calculator."""
        print(f"\n{'='*60}")
        print(f"F1 Environment Initialized: {self.gp}")
        print(f"{'='*60}")
        print(f"Predictor type: SimplifiedLaptimeCalculator")
        print(f"Using QMIP-compatible model (4 coefficients)")
        print(f"Expected: ~0.05us per prediction (100x faster than full model)")
        print(f"For {self.n_laps} laps: ~{self.n_laps * 0.05 / 1000:.2f}ms total prediction time")

        # Show calculator info
        info = self.laptime_calculator.get_info()
        print(f"\nModel coefficients:")
        print(f"  Intercept: {info['intercept']:.3f}s")
        print(f"  Compounds: {info['available_compounds']}")
        for compound in info['available_compounds']:
            comp_effect = info['compound_effects'][compound]
            tyre_deg = info['tyre_degradation'][compound]
            print(f"    C{compound}: base={comp_effect:+.3f}s, degradation={tyre_deg:+.6f}s/lap")
        print(f"  PitIn cost: {info['pit_in_cost']:.3f}s")
        print(f"  PitOut cost: {info['pit_out_cost']:.3f}s")
        print(f"{'='*60}\n")

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set random seed."""
        super().reset(seed=seed)
        return [seed]
