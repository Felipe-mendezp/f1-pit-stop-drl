"""
Step 3: Train yellow flag lap time models (RegLTYF - Eq. 3).

Full reproducible pipeline for the YF lap-time models consumed by the
simulation environment (trained_models/simulation/<GP>/laptime_yf_fast_v2.pkl):

1. Fit a per-GP OLS (no intercept) on 2021-2023 SC/VSC laps:
       LapTime ~ 0 + PitIn + PitOut + SC + SC_firstlap + VSC + VSC_firstlap
   including only the terms identifiable in that GP's data.
2. Some GPs lack SC or VSC history entirely (e.g. Belgian/Emilia Romagna
   have no VSC laps in 2021-2023): the missing coefficients are completed
   with SoftImpute (rank 2) over the GP x coefficient matrix of ALL GPs
   with YF data (Algorithm B.1 style completion).
3. For each target GP the completed coefficient vector is materialized as
   a FastLinearPredictor (exact-recovery synthetic OLS) and saved as
   laptime_yf_fast_v2.pkl (+ the statsmodels result as laptime_yf.pkl).
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
from tyrelife_2021 import adjust_2021


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
    adjust_2021(laps_2021)                       # factor de 2021: src/tyrelife_2021.py

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

YF_TERMS = ['PitIn', 'PitOut', 'SC', 'SC_firstlap', 'VSC', 'VSC_firstlap']

TARGET_GPS = [
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


def fit_all_gp_coefficients(df_yf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the per-GP YF OLS for EVERY GP with YF data and collect the
    coefficients into a (GP x term) matrix. Terms that are not identifiable
    in a GP (no such laps, or perfectly collinear, e.g. every SC lap is a
    first lap) are left as NaN for the completion step.
    """
    rows = {}
    for gp in sorted(df_yf['GP'].unique()):
        try:
            res = perform_linear_regression_yf(df_yf, gp, printear=False)
        except Exception as e:
            print(f"  Skipping {gp}: {e}")
            continue
        rows[gp] = {t: res.params.get(t, np.nan) for t in YF_TERMS}
    coef_df = pd.DataFrame.from_dict(rows, orient='index')[YF_TERMS]
    return coef_df


def complete_coefficients(coef_df: pd.DataFrame, max_rank: int = 2) -> pd.DataFrame:
    """
    Complete missing YF coefficients with SoftImpute over the GP x term
    matrix (Algorithm B.1 style). Rank-2 captures the dominant
    'circuit lap-time scale' factor plus one contrast.
    """
    from fancyimpute import SoftImpute

    n_missing = int(coef_df.isna().sum().sum())
    print(f"\nMatrix completion: {coef_df.shape[0]} GPs x {coef_df.shape[1]} terms, "
          f"{n_missing} missing entries")
    if n_missing == 0:
        return coef_df.copy()

    imputer = SoftImpute(
        max_rank=max_rank,
        init_fill_method="zero",
        max_iters=200,
        convergence_threshold=1e-6,
        verbose=False,
    )
    completed = imputer.fit_transform(coef_df.values)
    out = pd.DataFrame(completed, index=coef_df.index, columns=coef_df.columns)
    # Keep observed entries exactly (SoftImpute already does, but be explicit)
    out[~coef_df.isna()] = coef_df
    return out


def build_yf_predictor(coefs: Dict[str, float]) -> Tuple[object, FastLinearPredictor]:
    """
    Materialize a coefficient vector as a fitted statsmodels OLS result
    (and its FastLinearPredictor) via an exact-recovery synthetic design:
    y = X @ beta on a full-rank design reproduces beta exactly.
    """
    beta = np.array([coefs[t] for t in YF_TERMS], dtype=float)
    # Full-rank design covering each term (SC_firstlap/VSC_firstlap occur
    # together with SC/VSC, as in reality); duplicated for positive dof.
    X = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
    ], dtype=float)
    X = np.vstack([X, X])
    y = X @ beta
    df_synth = pd.DataFrame(X, columns=YF_TERMS)
    df_synth['LapTime'] = y
    formula = 'LapTime ~ 0 + ' + ' + '.join(YF_TERMS)
    res = smf.ols(formula=formula, data=df_synth).fit()
    assert np.allclose(res.params[YF_TERMS].values, beta), "exact recovery failed"
    return res, FastLinearPredictor(res, use_float32=True)


def train_yf_models(target_gps: list = None) -> Dict[str, Tuple]:
    """
    Train (and where necessary complete) the YF models for the target GPs
    and save them where ModelLoader expects them:
    trained_models/simulation/<GP>/laptime_yf_fast_v2.pkl (+ laptime_yf.pkl).
    """
    if target_gps is None:
        target_gps = TARGET_GPS

    print("Loading and preprocessing YF data...")
    df_yf = read_and_preprocess_yf_data()
    print(f"\nGPs with YF data (2021-2023): {df_yf['GP'].nunique()}")

    # 1-2. Per-GP fits + completion of non-identifiable coefficients
    coef_df = fit_all_gp_coefficients(df_yf)
    completed_df = complete_coefficients(coef_df, max_rank=2)

    # Persist the coefficient matrices for the paper appendix / audit
    coef_df.to_csv(str(config.SIMULATION_DIR / 'yf_coefficients_raw.csv'))
    completed_df.to_csv(str(config.SIMULATION_DIR / 'yf_coefficients_completed.csv'))

    # Sanity: SC slower than VSC, both slower than a green lap would be
    bad = completed_df[completed_df['SC'] <= completed_df['VSC']]
    if len(bad):
        print(f"WARNING: SC <= VSC after completion for: {list(bad.index)}")

    results = {}
    for gp in target_gps:
        if gp not in completed_df.index:
            print(f"\nWarning: {gp} has no YF row (no 2021-2023 YF laps at all). Skipping.")
            continue

        coefs = completed_df.loc[gp, YF_TERMS].to_dict()
        n_imputed = int(coef_df.loc[gp, YF_TERMS].isna().sum())
        print(f"\n{gp}: {coefs}  ({n_imputed} imputed)")

        sm_res, fast_pred = build_yf_predictor(coefs)

        gp_dir = config.SIMULATION_DIR / gp
        os.makedirs(gp_dir, exist_ok=True)
        with open(gp_dir / 'laptime_yf_fast_v2.pkl', 'wb') as f:
            pickle.dump(fast_pred, f)
        with open(gp_dir / 'laptime_yf.pkl', 'wb') as f:
            pickle.dump(sm_res, f)
        print(f"Saved: {gp_dir / 'laptime_yf_fast_v2.pkl'}")

        results[gp] = (sm_res, fast_pred)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — YF models saved for {len(results)} GPs")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    results = train_yf_models()
