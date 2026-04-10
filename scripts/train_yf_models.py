"""
Step 3: Train yellow flag lap time models (RegLTYF - Eq. 3).

Trains OLS models for SC/VSC lap times for target GPs,
saves both statsmodels and FastLinearPredictor versions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pickle
from typing import Dict, Tuple

from reg import FastLinearPredictor


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def read_and_preprocess_yf_data() -> pd.DataFrame:
    """
    Read and preprocess lap data to extract Yellow Flag (SC/VSC) laps.
    Uses data from 2021-2023 seasons.

    Returns:
        DataFrame containing only YF (SC/VSC) laps
    """

    def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
        """Remove wet/intermediate compounds and NaN values."""
        try:
            laps = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])]
        except Exception:
            pass
        laps = laps.dropna()
        return laps

    def pre_process_laps(laps: pd.DataFrame) -> pd.DataFrame:
        """
        Process TrackStatus to identify SC and VSC laps.
        TrackStatus codes: 1=GREEN, 2=YELLOW, 4=SC, 5=RED, 6=VSC, 7=VSC ENDING
        """
        # Drop rows with red flag (5)
        laps = laps[~laps['TrackStatus'].astype(str).str.contains('5')]
        laps['TrackStatus_2'] = laps['TrackStatus'].copy()

        # Convert combinations to base states
        # VSC ending (7) -> GREEN (1)
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({
            267.: 1., 67.: 1., 71.: 1., 7.: 1.
        })
        # Yellow + SC combinations -> SC (4)
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({24.: 4.})
        # VSC combinations -> VSC (6)
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({
            26.: 6., 126.: 6., 671.: 6., 1267.: 6., 167.: 6.,
            2671.: 6., 2167.: 6., 16.: 6.
        })
        # SC combinations -> SC (4)
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({
            124.: 4., 41.: 4., 14.: 4., 1264.: 4., 164.: 4.,
            214.: 4., 64.: 4., 264.: 4.
        })
        # Yellow flag combinations -> YELLOW (2)
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({12.: 2., 21.: 2.})

        # Remove yellow flag laps (only keep SC/VSC)
        laps = laps[laps['TrackStatus_2'] != 2.]

        # Binary SC and VSC variables
        laps['SC'] = (laps['TrackStatus_2'] == 4.).astype(int)
        laps['VSC'] = (laps['TrackStatus_2'] == 6.).astype(int)

        # Replace spaces in column names with underscores
        laps.columns = laps.columns.str.replace(' ', '_')
        return laps

    # Load data
    laps_2021 = pd.read_csv(str(config.DATA_DIR / 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(str(config.DATA_DIR / 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(str(config.DATA_DIR / 'session_2023_V2.csv'))

    # Apply tyre life adjustment for 2021 data
    tyrelife_adjust = {
        'C0': 0.65, 'C1': 1.0, 'C2': 0.90,
        'C3': 0.98, 'C4': 0.92, 'C5': 0.92
    }
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

    # Concatenate all years
    laps_all = pd.concat([laps_2021, laps_2022, laps_2023])

    # Filter wet/intermediate and NaN
    laps_all = filter_laps(laps_all)

    # Transform interval
    laps_all['Interval_front'] = np.exp(-laps_all['Interval_front'])

    # Process track status
    laps_all = pre_process_laps(laps_all)

    # Sort and create first lap indicators
    laps_all = laps_all.sort_values(["Year", "GP", "Driver", "LapNumber"]).copy()
    g = laps_all.groupby(["Year", "GP", "Driver"], sort=False)

    # First lap of VSC/SC segment
    laps_all["VSC_firstlap"] = (
        (laps_all["VSC"].eq(1)) &
        (g["VSC"].shift(1).fillna(0).eq(0))
    ).astype(int)

    laps_all["SC_firstlap"] = (
        (laps_all["SC"].eq(1)) &
        (g["SC"].shift(1).fillna(0).eq(0))
    ).astype(int)

    # Filter to only YF laps (TrackStatus != 1)
    laps_yf = laps_all[laps_all['TrackStatus_2'] != 1.]

    return laps_yf


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def perform_linear_regression_yf(laps: pd.DataFrame, gp: str,
                                  printear: bool = True) -> smf.ols:
    """
    Perform linear regression for Yellow Flag lap times for a specific GP.

    The model uses no intercept and includes:
    - SC/VSC indicators
    - First lap of SC/VSC indicators
    - PitIn/PitOut indicators

    Args:
        laps: DataFrame with YF lap data
        gp: Grand Prix name
        printear: Whether to print summary

    Returns:
        Fitted statsmodels OLS result
    """
    df_laps = laps[laps['GP'] == gp].copy()

    if len(df_laps) == 0:
        raise ValueError(f"No YF data found for GP: {gp}")

    print(f"\n{'='*60}")
    print(f"Training YF model for: {gp}")
    print(f"Number of YF laps: {len(df_laps)}")
    print(f"SC laps: {df_laps['SC'].sum()}, VSC laps: {df_laps['VSC'].sum()}")

    # Build formula dynamically based on available data
    formula = 'LapTime ~ 0'  # No intercept

    # Add PitIn if present
    if df_laps['PitIn'].max() > 0:
        formula += ' + PitIn'

    # Add PitOut if present
    if df_laps['PitOut'].max() > 0:
        formula += ' + PitOut'

    # Add SC if present
    if df_laps['SC'].sum() > 0:
        formula += ' + SC'
        # Add SC_firstlap if there are non-first SC laps
        if df_laps['SC_firstlap'].sum() < df_laps['SC'].sum():
            formula += ' + SC_firstlap'

    # Add VSC if present
    if df_laps['VSC'].sum() > 0:
        formula += ' + VSC'
        # Add VSC_firstlap if there are non-first VSC laps
        if df_laps['VSC_firstlap'].sum() < df_laps['VSC'].sum():
            formula += ' + VSC_firstlap'

    print(f"Formula: {formula}")

    # Fit model
    model = smf.ols(formula=formula, data=df_laps)
    results = model.fit()

    # Check matrix rank
    X = results.model.exog
    rank = np.linalg.matrix_rank(X)
    n_cols = X.shape[1]

    if rank < n_cols:
        print("Warning: Design matrix has linear dependence")

    if printear:
        print(results.summary())

    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_missing_yf_models(target_gps: list = None) -> Dict[str, Tuple]:
    """
    Train YF models for specified GPs.

    Args:
        target_gps: List of GP names to train. If None, trains Belgian, Dutch,
                    Emilia Romagna, Miami

    Returns:
        Dictionary with GP name -> (statsmodels_result, FastLinearPredictor)
    """
    if target_gps is None:
        target_gps = [
            'Belgian Grand Prix',
            'Dutch Grand Prix',
            'Emilia Romagna Grand Prix',
            'Miami Grand Prix'
        ]

    # Load and preprocess data
    print("Loading and preprocessing YF data...")
    df_yf = read_and_preprocess_yf_data()

    available_gps = df_yf['GP'].unique()
    print(f"\nAvailable GPs with YF data: {len(available_gps)}")

    results = {}
    models_dir = str(config.SIMULATION_DIR)
    os.makedirs(models_dir, exist_ok=True)

    for gp in target_gps:
        if gp not in available_gps:
            print(f"\nWarning: {gp} not found in YF data. Skipping...")
            continue

        try:
            # Train model
            model = perform_linear_regression_yf(df_yf, gp, printear=True)

            # Create FastLinearPredictor for fast inference
            fast_pred = FastLinearPredictor(model, use_float32=True)

            # Save statsmodels model
            filepath_sm = f'{models_dir}/{gp}_YF_reg_Pickle.pkl'
            with open(filepath_sm, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved statsmodels model: {filepath_sm}")

            # Save FastLinearPredictor
            filepath_fast = f'{models_dir}/{gp}_YF_fast_predictor.pkl'
            with open(filepath_fast, 'wb') as f:
                pickle.dump(fast_pred, f)
            print(f"Saved FastLinearPredictor: {filepath_fast}")

            results[gp] = (model, fast_pred)

        except Exception as e:
            print(f"\nError training model for {gp}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Successfully trained models for {len(results)} GPs:")
    for gp in results:
        print(f"  - {gp}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    results = train_missing_yf_models()
