# Unified ModelLoader for F1 Multi-Driver Simulation Environment
# Loads and manages all models required for simulation

import os
import json
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

# Import FastLinearPredictor for type hints and loading
from reg import FastLinearPredictor

# Make FastLinearPredictor available in __main__ for pickle compatibility
if '__main__' in sys.modules:
    sys.modules['__main__'].FastLinearPredictor = FastLinearPredictor


@dataclass
class GPModels:
    """
    Container for all models needed for a GP simulation.

    This dataclass holds all the predictive models and configuration
    parameters required to run a multi-driver F1 race simulation.

    Attributes:
        gp_name: Name of the Grand Prix
        n_laps: Total number of laps in the race
        compounds: List of available tire compounds (e.g., ['C2', 'C3', 'C4'])
        vsc_prob: Probability of VSC per lap
        sc_prob: Probability of SC per lap

        laptime_clear: FastLinearPredictor for clear track lap times
        laptime_yf: Statsmodels result for SC/VSC lap times

        overtake_logit: Statsmodels logit result for overtake probability
        overtake_threshold: Threshold for overtake prediction

        rival_pitstop_logit: Statsmodels logit for pit stop prediction (handles clear + YF)
        rival_compound_logit: Statsmodels logit for compound choice prediction
    """
    # GP metadata
    gp_name: str
    n_laps: int
    compounds: List[str]
    vsc_prob: float
    sc_prob: float

    # Lap time models
    laptime_clear: FastLinearPredictor
    laptime_yf: Any  # statsmodels result

    # Overtake model
    overtake_logit: Any  # statsmodels logit result
    overtake_threshold: float = 0.5

    # Rival strategy models (full logit models)
    rival_pitstop_logit: Any = None  # statsmodels logit for pit stop (clear + YF)
    rival_compound_logit: Any = None  # statsmodels logit for compound choice

    def get_compound_index(self, compound: str) -> int:
        """Get the index of a compound in the compounds list."""
        try:
            return self.compounds.index(compound)
        except ValueError:
            raise ValueError(f"Compound {compound} not in {self.compounds}")

    def __repr__(self) -> str:
        return (f"GPModels(gp_name='{self.gp_name}', n_laps={self.n_laps}, "
                f"compounds={self.compounds})")


class ModelLoader:
    """
    Loads and manages all simulation models for F1 environment.

    This class provides a unified interface to load all models required
    for the multi-driver F1 simulation environment. Models are cached
    after first load for efficiency.

    Usage:
        loader = ModelLoader()
        models = loader.load('Bahrain Grand Prix')
        env = create_f1_env_from_models(models, driver='VER', ...)

    Attributes:
        models_path: Path to the Models/simulation directory
        _cache: Dictionary caching loaded GPModels
    """

    SUPPORTED_GPS = [
        'Bahrain Grand Prix',
        'Belgian Grand Prix',
        'Dutch Grand Prix',
        'Emilia Romagna Grand Prix',
        'Hungarian Grand Prix',
        'Miami Grand Prix',
        'Saudi Arabian Grand Prix',
        'Singapore Grand Prix',
        'United States Grand Prix'
    ]

    def __init__(self, models_path: Optional[str] = None):
        """
        Initialize the ModelLoader.

        Args:
            models_path: Path to Models/simulation directory.
                        If None, uses default path.
        """
        if models_path is None:
            from config import SIMULATION_DIR
            models_path = str(SIMULATION_DIR)
        self.models_path = models_path
        self._cache: Dict[str, GPModels] = {}

    def _load_pickle(self, filepath: str) -> Any:
        """Load a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _load_json(self, filepath: str) -> Dict:
        """Load a JSON config file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def load(self, gp: str, use_cache: bool = True) -> GPModels:
        """
        Load all models for a Grand Prix.

        Args:
            gp: Grand Prix name (e.g., 'Bahrain Grand Prix')
            use_cache: Whether to use cached models if available

        Returns:
            GPModels container with all models

        Raises:
            ValueError: If GP is not supported
            FileNotFoundError: If required model files are missing
        """
        if gp not in self.SUPPORTED_GPS:
            raise ValueError(
                f"GP '{gp}' not supported. "
                f"Supported GPs: {self.SUPPORTED_GPS}"
            )

        # Check cache
        if use_cache and gp in self._cache:
            return self._cache[gp]

        gp_dir = os.path.join(self.models_path, gp)

        if not os.path.exists(gp_dir):
            raise FileNotFoundError(f"GP directory not found: {gp_dir}")

        # Load config
        config = self._load_json(os.path.join(gp_dir, 'config.json'))

        # Load models
        laptime_clear = self._load_pickle(
            os.path.join(gp_dir, 'laptime_clear.pkl')
        )
        # Prefer v2 (all 6 YF coefficients via matrix completion),
        # then v1 FastLinearPredictor, then raw statsmodels pickle
        yf_v2_path   = os.path.join(gp_dir, 'laptime_yf_fast_v2.pkl')
        yf_fast_path = os.path.join(gp_dir, 'laptime_yf_fast.pkl')
        if os.path.exists(yf_v2_path):
            laptime_yf = self._load_pickle(yf_v2_path)
        elif os.path.exists(yf_fast_path):
            laptime_yf = self._load_pickle(yf_fast_path)
        else:
            laptime_yf = self._load_pickle(
                os.path.join(gp_dir, 'laptime_yf.pkl')
            )
        overtake_logit = self._load_pickle(
            os.path.join(gp_dir, 'overtake_logit.pkl')
        )

        # Load overtake threshold (use default if not available)
        try:
            overtake_threshold = self._load_pickle(
                os.path.join(gp_dir, 'overtake_threshold.pkl')
            )
        except FileNotFoundError:
            overtake_threshold = 0.5

        # Load rival logit models
        rival_pitstop_logit = self._load_pickle(
            os.path.join(gp_dir, 'rival_pitstop_logit.pkl')
        )
        rival_compound_logit = self._load_pickle(
            os.path.join(gp_dir, 'rival_compound_logit.pkl')
        )

        # Create GPModels container
        models = GPModels(
            gp_name=config['gp_name'],
            n_laps=config['n_laps'],
            compounds=config['compounds'],
            vsc_prob=config['vsc_prob'],
            sc_prob=config['sc_prob'],
            laptime_clear=laptime_clear,
            laptime_yf=laptime_yf,
            overtake_logit=overtake_logit,
            overtake_threshold=overtake_threshold,
            rival_pitstop_logit=rival_pitstop_logit,
            rival_compound_logit=rival_compound_logit,
        )

        # Cache
        self._cache[gp] = models

        return models

    def load_all(self) -> Dict[str, GPModels]:
        """
        Load models for all supported GPs.

        Returns:
            Dictionary mapping GP name to GPModels
        """
        all_models = {}
        for gp in self.SUPPORTED_GPS:
            try:
                all_models[gp] = self.load(gp)
            except Exception as e:
                print(f"Warning: Failed to load {gp}: {e}")
        return all_models

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()

    def is_loaded(self, gp: str) -> bool:
        """Check if a GP's models are cached."""
        return gp in self._cache

    def get_supported_gps(self) -> List[str]:
        """Get list of supported GP names."""
        return self.SUPPORTED_GPS.copy()

    def validate_gp(self, gp: str) -> Dict[str, bool]:
        """
        Validate that all required files exist for a GP.

        Args:
            gp: Grand Prix name

        Returns:
            Dictionary with file name -> exists status
        """
        required_files = [
            'config.json',
            'laptime_clear.pkl',
            'laptime_yf.pkl',
            'overtake_logit.pkl',
            'rival_pitstop_logit.pkl',
            'rival_compound_logit.pkl',
        ]

        gp_dir = os.path.join(self.models_path, gp)
        status = {}

        for fname in required_files:
            fpath = os.path.join(gp_dir, fname)
            status[fname] = os.path.exists(fpath)

        return status

    def validate_all(self) -> Dict[str, Dict[str, bool]]:
        """
        Validate all supported GPs.

        Returns:
            Dictionary mapping GP name to validation status
        """
        return {gp: self.validate_gp(gp) for gp in self.SUPPORTED_GPS}


# Convenience function for quick loading
def load_gp_models(gp: str) -> GPModels:
    """
    Convenience function to load models for a single GP.

    Args:
        gp: Grand Prix name

    Returns:
        GPModels container
    """
    loader = ModelLoader()
    return loader.load(gp)


# Test function
def test_model_loader():
    """Test the ModelLoader by loading all GPs."""
    print("="*70)
    print("Testing ModelLoader")
    print("="*70)

    loader = ModelLoader()

    # Validate all GPs
    print("\n1. Validating all GPs:")
    validation = loader.validate_all()
    for gp, status in validation.items():
        all_ok = all(status.values())
        symbol = "OK" if all_ok else "INCOMPLETE"
        print(f"  {gp}: {symbol}")
        if not all_ok:
            missing = [k for k, v in status.items() if not v]
            print(f"    Missing: {missing}")

    # Load all GPs
    print("\n2. Loading all GPs:")
    all_models = loader.load_all()
    print(f"  Successfully loaded {len(all_models)} GPs")

    # Test each loaded model
    print("\n3. Testing loaded models:")
    for gp, models in all_models.items():
        print(f"\n  {gp}:")
        print(f"    Laps: {models.n_laps}")
        print(f"    Compounds: {models.compounds}")
        print(f"    VSC/SC prob: {models.vsc_prob:.2f}/{models.sc_prob:.2f}")
        print(f"    Has pitstop logit: {models.rival_pitstop_logit is not None}")
        print(f"    Has compound logit: {models.rival_compound_logit is not None}")

        # Test laptime prediction
        test_data = {
            'LapNumber': 10,
            'LapNumber_2': 100,
            'TyreLife': 5,
            'Interval_front': 0.9,
            'Interval_behind': 0.8,
            'PitIn': 0,
            'PitOut': 0,
            'DRS': 0,
            'FirstLap_pos': 0,
            'Driver': 'VER',
            'Compound_Detail': models.compounds[0],
            'Year': '2023'
        }
        pred = models.laptime_clear.predict_single(test_data)
        print(f"    Test laptime prediction: {pred:.2f}s")

    print("\n" + "="*70)
    print("ModelLoader test complete!")
    print("="*70)


if __name__ == "__main__":
    test_model_loader()
