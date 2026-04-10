"""
Step 4: Complete missing tire coefficients via SoftImpute.

Runs the matrix completion pipeline (fun_real_data_v2) from
models.matrix_completion which:
  - Loads regression coefficients from compounds_coefs_V3.xlsx
  - Applies compatibility checks and offset normalization
  - Imputes missing coefficients with SoftImpute (rank-1)
  - Saves completed_coefs_V2_filled.xlsx to DATA_DIR
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

from models.matrix_completion import fun_real_data_v2


def main():
    print("=" * 70)
    print("MATRIX COMPLETION: Completing missing tire coefficients")
    print("=" * 70)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Output will be saved to: {config.DATA_DIR / 'completed_coefs_V2_filled.xlsx'}")
    print()

    fun_real_data_v2()

    print("\nMatrix completion finished.")


if __name__ == "__main__":
    main()
