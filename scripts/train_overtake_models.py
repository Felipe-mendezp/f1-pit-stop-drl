"""
Step 5: Train overtake logit models.

Wrapper that calls train_missing_overtake_models() from
models.overtake_models to train logistic regression models
for predicting overtake probability.

Target GPs: Emilia Romagna, Miami (default).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

from models.overtake_models import train_missing_overtake_models


def main():
    print("=" * 70)
    print("OVERTAKE MODELS: Training logit models for overtake prediction")
    print("=" * 70)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Models directory: {config.SIMULATION_DIR}")
    print()

    results = train_missing_overtake_models()

    print(f"\nReturned results for {len(results)} GPs")


if __name__ == "__main__":
    main()
