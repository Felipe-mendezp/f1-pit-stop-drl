"""
Step 6: Train rival decision models.

Wrapper that calls train_all_models() from models.rival_models
to train:
  1. Pit stop decision logit (normal racing)
  2. Pit stop decision logit (yellow flag)
  3. Compound choice conditional logit

Per-GP coefficient arrays are extracted and saved for the RL environment.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

from models.rival_models import train_all_models


def main():
    print("=" * 70)
    print("RIVAL MODELS: Training pit stop / compound choice models")
    print("=" * 70)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Models directory: {config.SIMULATION_DIR}")
    print()

    # Define GPs to train (only the selected GPs from the paper)
    SELECTED_GP = {
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

    train_all_models(SELECTED_GP)

    print("\nRival model training finished.")


if __name__ == "__main__":
    main()
