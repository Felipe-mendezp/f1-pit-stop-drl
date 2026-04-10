"""
Script to train and save rival decision models for F1 environment V3.

This script trains:
1. Pit stop decision logit (normal racing)
2. Pit stop decision logit (yellow flag)
3. Compound choice conditional logit

Models are saved as pickle files.
Run this script once before training agents.
"""

import config

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle
import os
from typing import Dict, Tuple, List

# === CONSTANTS ===

DATA_PATH = str(config.DATA_DIR)
MODELS_PATH = str(config.SIMULATION_DIR)

LAPS_DD = {
    'Abu Dhabi Grand Prix': 58,
    'Australian Grand Prix': 58,
    'Bahrain Grand Prix': 57,
    'Belgian Grand Prix': 44,
    'Canadian Grand Prix': 70,
    'Dutch Grand Prix': 72,
    'Hungarian Grand Prix': 70,
    'Italian Grand Prix': 53,
    'Mexico City Grand Prix': 71,
    'Saudi Arabian Grand Prix': 50,
    'Singapore Grand Prix': 62,
    'São Paulo Grand Prix': 69,
    'United States Grand Prix': 56,
    'Emilia Romagna Grand Prix': 63,
    'Miami Grand Prix': 57,
}

COMPOUND_MAPS = {
    2021: {
        'Bahrain Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Portuguese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'French Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Styrian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Hungarian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Italian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Russian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Turkish Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    2022: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'French Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Hungarian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Italian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    2023: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    2024: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'São Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    }
}

TYRELIFE_ADJUST = {
    'C0': 0.65,
    'C1': 1.0,
    'C2': 0.90,
    'C3': 0.98,
    'C4': 0.92,
    'C5': 0.92
}


# === DATA LOADING ===

def load_lap_data() -> pd.DataFrame:
    """Load and concatenate lap data from 2021-2023."""
    laps_2021 = pd.read_csv(f'{DATA_PATH}/session_2021_V2.csv')
    laps_2022 = pd.read_csv(f'{DATA_PATH}/session_2022_V2.csv')
    laps_2023 = pd.read_csv(f'{DATA_PATH}/session_2023_V2.csv')

    # Adjust TyreLife for 2021 data
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(TYRELIFE_ADJUST)

    # Concatenate all years
    laps = pd.concat([laps_2021, laps_2022, laps_2023], ignore_index=True)

    return laps


def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Filter out wet/intermediate compounds and NA values."""
    laps = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])]
    laps = laps.dropna()
    return laps


def add_used_two_compounds(laps: pd.DataFrame) -> pd.DataFrame:
    """Add column indicating if driver has used at least two different compounds."""
    laps = laps.sort_values(['Driver', 'GP', 'Year', 'LapNumber'])

    # Mark first time each compound appears in a stint
    laps['is_new_compound'] = (
        laps.groupby(['Driver', 'GP', 'Year', 'Compound_Detail'])
            .cumcount()
            .eq(0)
            .astype(int)
    )

    # Cumulative count of unique compounds
    laps['unique_compounds_so_far'] = (
        laps.groupby(['Driver', 'GP', 'Year'])['is_new_compound']
            .cumsum()
    )

    # Number of unique compounds BEFORE this lap
    unique_before = (
        laps.groupby(['Driver', 'GP', 'Year'])['unique_compounds_so_far']
            .shift(fill_value=0)
    )

    # Final column: 1 if already used at least 2 compounds
    laps['used_two_compounds'] = (unique_before >= 2).astype(int)

    return laps


def preprocess_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for lap data."""
    laps = filter_laps(laps)
    laps = add_used_two_compounds(laps)

    # Transform intervals
    laps['Interval_front_real'] = laps['Interval_front'].copy()
    laps['Interval_front'] = np.exp(-laps['Interval_front'])
    laps['Interval_behind'] = np.exp(-laps['Interval_behind'])

    # DRS indicator
    laps['DRS'] = (
        (laps['Interval_front_real'] < 1.0) &
        (laps['LapNumber'] > 2)
    ).astype(int)

    # Squared lap number
    laps['LapNumber_2'] = laps['LapNumber'] ** 2

    # First lap position
    first_lap = laps['LapNumber'] == 1.0
    laps['FirstLap_pos'] = laps['Position'] * first_lap

    # Process TrackStatus
    laps = preprocess_track_status(laps)

    # Add LapsLeft
    laps['LapsLeft'] = laps.apply(
        lambda row: LAPS_DD.get(row['GP'], 0) - row['LapNumber'], axis=1
    )

    # Add EarlyLaps and LateLaps (binary race phase indicators)
    laps['EarlyLaps'] = (laps['LapNumber'] <= 10).astype(int)
    laps['LateLaps'] = laps.apply(
        lambda row: 1 if row['LapNumber'] >= (LAPS_DD.get(row['GP'], 0) - 10) else 0, axis=1
    )

    # NewCompound (compound after pit stop)
    laps = laps.sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
    laps['NewCompound'] = (
        laps.groupby(['Year', 'GP', 'Driver'])['Compound_Detail']
            .shift(-1)
    )
    laps.loc[laps['PitIn'] == 0, 'NewCompound'] = np.nan

    # Replace spaces in column names
    laps.columns = laps.columns.str.replace(' ', '_')

    return laps


def preprocess_track_status(laps: pd.DataFrame) -> pd.DataFrame:
    """Process TrackStatus to create SC and VSC indicators."""
    # Remove red flag laps
    laps = laps[~laps['TrackStatus'].astype(str).str.contains('5')]

    laps['TrackStatus_2'] = laps['TrackStatus'].copy()

    # VSC ending -> Green
    laps['TrackStatus_2'] = laps['TrackStatus_2'].replace(
        {267.: 1., 67.: 1., 71.: 1., 7.: 1.}
    )
    # Various SC combinations -> SC
    laps['TrackStatus_2'] = laps['TrackStatus_2'].replace(
        {24.: 4., 124.: 4., 41.: 4., 14.: 4., 1264.: 4., 164.: 4.,
         214.: 4., 64.: 4., 264.: 4.}
    )
    # Various VSC combinations -> VSC
    laps['TrackStatus_2'] = laps['TrackStatus_2'].replace(
        {26.: 6., 126.: 6., 671.: 6., 1267.: 6., 167.: 6.,
         2671.: 6., 2167.: 6., 16.: 6.}
    )
    # Yellow flag -> remove
    laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({12.: 2., 21.: 2.})
    laps = laps[laps['TrackStatus_2'] != 2.]

    # Binary indicators
    laps['SC'] = (laps['TrackStatus_2'] == 4.).astype(int)
    laps['VSC'] = (laps['TrackStatus_2'] == 6.).astype(int)

    return laps


# === MODEL TRAINING ===

def train_pit_stop_logit(laps: pd.DataFrame, print_summary: bool = True) -> Tuple:
    """
    Train logit model for pit stop decision during normal racing.

    Formula: PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP

    Returns:
        Tuple of (model result, optimal threshold)
    """
    formula = "PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP"

    model = smf.logit(formula=formula, data=laps)
    result = model.fit(disp=False)

    if print_summary:
        print("\n" + "="*60)
        print("PIT STOP DECISION MODEL (Normal Racing)")
        print("="*60)
        print(result.summary())

    # Calculate optimal threshold
    predicted_probs = result.predict(laps)
    n_pitins = laps['PitIn'].sum()
    sorted_probs = np.sort(predicted_probs)[::-1]
    threshold = sorted_probs[int(n_pitins) - 1]

    print(f"\nOptimal threshold: {threshold:.4f}")
    print(f"Total pit stops: {int(n_pitins)}")

    return result, threshold


def train_pit_stop_logit_yf(laps: pd.DataFrame, print_summary: bool = True) -> Tuple:
    """
    Train logit model for pit stop decision during yellow flag (SC/VSC).

    Formula: PitIn ~ SC + TyreLife + LapsLeft + Position + TyreLife:C(Compound_Detail) + LapsLeft:C(Compound_Detail)

    Returns:
        Tuple of (model result, optimal threshold)
    """
    # Filter for yellow flag laps only
    laps_yf = laps[(laps['SC'] == 1) | (laps['VSC'] == 1)].copy()

    if len(laps_yf) < 50:
        print("Warning: Not enough yellow flag data for robust model")
        return None, 0.5

    # Create interaction terms
    for c in ['C1', 'C2', 'C3', 'C4', 'C5']:
        compound_mask = (laps_yf['Compound_Detail'] == c).astype(float)
        laps_yf[f'TyreLife_x_Compound_{c}'] = laps_yf['TyreLife'] * compound_mask
        laps_yf[f'LapsLeft_x_Compound_{c}'] = laps_yf['LapsLeft'] * compound_mask

    # Build features
    feature_cols = ['SC', 'TyreLife', 'LapsLeft', 'Position']
    feature_cols += [f'TyreLife_x_Compound_{c}' for c in ['C1', 'C2', 'C3', 'C4', 'C5']]
    feature_cols += [f'LapsLeft_x_Compound_{c}' for c in ['C1', 'C2', 'C3', 'C4', 'C5']]

    X = laps_yf[feature_cols].copy()
    X = sm.add_constant(X)
    y = laps_yf['PitIn']

    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=1000)

    if print_summary:
        print("\n" + "="*60)
        print("PIT STOP DECISION MODEL (Yellow Flag)")
        print("="*60)
        print(result.summary())

    # Calculate optimal threshold
    predicted_probs = result.predict(X)
    n_pitins = y.sum()
    if n_pitins > 0:
        sorted_probs = np.sort(predicted_probs)[::-1]
        threshold = sorted_probs[int(n_pitins) - 1]
    else:
        threshold = 0.5

    print(f"\nOptimal threshold: {threshold:.4f}")
    print(f"Yellow flag pit stops: {int(n_pitins)}")

    return result, threshold


def train_compound_choice_logit(laps: pd.DataFrame, print_summary: bool = True):
    """
    Train conditional logit model for compound choice when pitting.

    Formula: chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP

    Returns:
        Fitted model result
    """
    # Build dataframe of available compounds per GP/Year
    rows = []
    for year, gps in COMPOUND_MAPS.items():
        for gp_name, comp_map in gps.items():
            for role, compound in comp_map.items():
                rows.append({
                    'Year': year,
                    'GP': gp_name,
                    'Tyre_role': role,
                    'alt': compound
                })
    df_compounds_long = pd.DataFrame(rows)

    # Filter to pit stops with known new compound
    df = laps.copy()
    df = df[(df['PitIn'] == 1) & df['NewCompound'].notna()].copy()

    if len(df) < 50:
        print("Warning: Not enough pit stop data for compound choice model")
        return None

    # Create decision ID
    df['decision_id'] = (
        df['Year'].astype(str) + '_' +
        df['GP'].astype(str) + '_' +
        df['Driver'].astype(str) + '_' +
        df['LapNumber'].astype(int).astype(str)
    )

    # Merge with available compounds
    df_long = df.merge(df_compounds_long, on=['Year', 'GP'], how='left')

    # Mark chosen alternative
    df_long['chosen'] = (df_long['alt'] == df_long['NewCompound']).astype(int)

    # Not change compound indicator
    df_long['not_change_compound'] = (
        (df_long['alt'] == df_long['Compound_Detail']) &
        (df_long['used_two_compounds'] == 0)
    ).astype(int)

    # Fit conditional logit
    formula = "chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP"

    model = smf.logit(formula=formula, data=df_long)
    result = model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df_long['decision_id']},
        disp=False
    )

    if print_summary:
        print("\n" + "="*60)
        print("COMPOUND CHOICE MODEL (Conditional Logit)")
        print("="*60)
        print(result.summary())

    return result


# === COEFFICIENT EXTRACTION ===

def get_gp_compounds(year: int, gp: str) -> List[str]:
    """Get available compounds for a GP in a given year."""
    compound_map = COMPOUND_MAPS.get(year, {})
    gp_compounds = compound_map.get(gp, {})
    gp_compounds = dict(sorted(gp_compounds.items(), key=lambda item: item[1]))
    return list(gp_compounds.values())


def extract_stop_betas(result, gp: str, gp_compounds: List[str]) -> np.ndarray:
    """
    Extract coefficient array for pit stop prediction.

    Order: [Intercept, SC, VSC, used_two_compounds, EarlyLaps, LateLaps,
            TyreLife:C1, TyreLife:C2, ..., Position, GP]
    """
    params = result.params

    coef_intercept = params['Intercept']
    coef_SC = params['SC']
    coef_VSC = params['VSC']
    coef_used_two = params['used_two_compounds']
    coef_EarlyLaps = params['EarlyLaps']
    coef_LateLaps = params['LateLaps']

    # TyreLife:Compound interaction terms
    coef_TyreLife = []
    for compound in gp_compounds:
        param_name = f'TyreLife:C(Compound_Detail)[{compound}]'
        if param_name in params.index:
            coef_TyreLife.append(params[param_name])
        else:
            coef_TyreLife.append(0.0)

    coef_position = params['Position']

    # GP dummy
    gp_param = f'GP[T.{gp}]'
    coef_GP = params.get(gp_param, 0.0)

    coef_array = np.concatenate([
        np.array([coef_intercept, coef_SC, coef_VSC, coef_used_two, coef_EarlyLaps, coef_LateLaps]),
        np.array(coef_TyreLife),
        np.array([coef_position, coef_GP])
    ])

    return coef_array.astype(np.float32)


def extract_compound_betas(result, gp: str, gp_compounds: List[str]) -> np.ndarray:
    """
    Extract 2D coefficient array for compound choice prediction.

    Shape: (n_compounds, 4)
    Columns: [Intercept(alt), GP, LapsLeft(alt), not_change_compound]
    """
    params = result.params

    coef_intercepts = np.array([
        params[f"C(alt)[{c}]"] for c in gp_compounds
    ])

    gp_param = f"GP[T.{gp}]"
    coef_GP = np.array([
        params.get(gp_param, 0.0) for _ in gp_compounds
    ])

    coef_LapsLeft = np.array([
        params.get(f"C(alt)[{c}]:LapsLeft", 0.0) for c in gp_compounds
    ])

    coef_not_change = np.array([
        params.get("not_change_compound", 0.0) for _ in gp_compounds
    ])

    coef_2D = np.vstack([coef_intercepts, coef_GP, coef_LapsLeft, coef_not_change]).T

    return coef_2D.astype(np.float32)


# === MAIN TRAINING FUNCTION ===

def train_all_models(selected_gps: Dict[str, List[str]], year: int = 2024):
    """
    Train all rival models and save to disk.

    Args:
        selected_gps: Dictionary mapping GP names to available compounds
        year: Year for extracting compound information
    """
    # Ensure output directory exists
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("Loading and preprocessing data...")
    laps = load_lap_data()
    laps = preprocess_laps(laps)

    print(f"\nTotal laps after preprocessing: {len(laps)}")
    print(f"Unique GPs: {laps['GP'].nunique()}")

    # Train global models
    print("\n" + "="*70)
    print("TRAINING GLOBAL MODELS")
    print("="*70)

    result_stop, threshold_stop = train_pit_stop_logit(laps)
    result_stop_yf, threshold_stop_yf = train_pit_stop_logit_yf(laps)
    result_compound = train_compound_choice_logit(laps)

    # Save global models
    with open(f'{MODELS_PATH}/global_logit_stop.pkl', 'wb') as f:
        pickle.dump(result_stop, f)
    with open(f'{MODELS_PATH}/global_logit_stop_threshold.pkl', 'wb') as f:
        pickle.dump(threshold_stop, f)

    if result_stop_yf is not None:
        with open(f'{MODELS_PATH}/global_logit_stop_yf.pkl', 'wb') as f:
            pickle.dump(result_stop_yf, f)
        with open(f'{MODELS_PATH}/global_logit_stop_yf_threshold.pkl', 'wb') as f:
            pickle.dump(threshold_stop_yf, f)

    if result_compound is not None:
        with open(f'{MODELS_PATH}/global_logit_compound.pkl', 'wb') as f:
            pickle.dump(result_compound, f)

    # Extract and save per-GP coefficients
    print("\n" + "="*70)
    print("EXTRACTING GP-SPECIFIC COEFFICIENTS")
    print("="*70)

    for gp, compounds in selected_gps.items():
        print(f"\nProcessing: {gp}")
        print(f"  Compounds: {compounds}")

        try:
            # Extract stop coefficients
            betas_stop = extract_stop_betas(result_stop, gp, compounds)
            print(f"  Stop betas shape: {betas_stop.shape}")

            with open(f'{MODELS_PATH}/{gp}_betas_stop.pkl', 'wb') as f:
                pickle.dump(betas_stop, f)

            # Extract compound choice coefficients
            if result_compound is not None:
                betas_compound = extract_compound_betas(result_compound, gp, compounds)
                print(f"  Compound betas shape: {betas_compound.shape}")

                with open(f'{MODELS_PATH}/{gp}_betas_compound.pkl', 'wb') as f:
                    pickle.dump(betas_compound, f)

            # Save threshold
            with open(f'{MODELS_PATH}/{gp}_threshold_stop.pkl', 'wb') as f:
                pickle.dump(threshold_stop, f)

            if result_stop_yf is not None:
                with open(f'{MODELS_PATH}/{gp}_threshold_stop_yf.pkl', 'wb') as f:
                    pickle.dump(threshold_stop_yf, f)

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModels saved to: {MODELS_PATH}")


# === ENTRY POINT ===

if __name__ == "__main__":
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
