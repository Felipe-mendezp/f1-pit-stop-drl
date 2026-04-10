# Script to train missing overtake logit models
# Target GPs: Emilia Romagna, Miami

import config

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import statsmodels.api as sm
import pickle
from typing import Dict, Any

from reg import FastLinearPredictor


def load_and_prepare_data():
    """
    Load lap data and prepare it for overtake analysis.
    """
    # Load data
    laps_2021 = pd.read_csv(str(config.DATA_DIR / 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(str(config.DATA_DIR / 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(str(config.DATA_DIR / 'session_2023_V2.csv'))

    # Apply tyre life adjustment for 2021
    tyrelife_adjust = {
        'C0': 0.65, 'C1': 1.0, 'C2': 0.90,
        'C3': 0.98, 'C4': 0.92, 'C5': 0.92
    }
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

    # Target GPs
    selected_GP = {
        'Emilia Romagna Grand Prix': ['C3', 'C4', 'C5'],
        'Miami Grand Prix': ['C2', 'C3', 'C4']
    }

    return laps_2021, laps_2022, laps_2023, selected_GP


def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Filter out wet/intermediate compounds and C0."""
    laps_filter = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])].copy()
    laps_filter = laps_filter[laps_filter['Compound_Detail'].isin(['C1', 'C2', 'C3', 'C4', 'C5'])].copy()

    # Calculate laps left
    max_laps = laps_filter.groupby(['Year', 'GP', 'Driver'])['LapNumber'].transform('max')
    laps_filter['LapsLeft'] = max_laps - laps_filter['LapNumber']

    return laps_filter


def prepare_data_for_logit(laps_2021, laps_2022, laps_2023, selected_GP, reg_clear_dd):
    """
    Prepare data for overtake logit modeling.
    """
    # Concatenate
    laps_all = pd.concat([laps_2021, laps_2022, laps_2023])
    laps_all = filter_laps(laps_all)

    # Save original interval before transformation
    laps_all['interval_front_real'] = laps_all['Interval_front'].copy()

    # Calculate DRS
    laps_all['DRS'] = (
        (laps_all['Interval_front'] < 1.0) &
        (laps_all['LapNumber'] > 2)
    ).astype(int)

    # Transform intervals
    laps_all['Interval_front'] = np.exp(-laps_all['Interval_front'])
    laps_all['Interval_behind'] = np.exp(-laps_all['Interval_behind'])
    laps_all['LapNumber_2'] = laps_all['LapNumber'] ** 2

    # First lap position
    first_lap = laps_all['LapNumber'] == 1.0
    laps_all['FirstLap_pos'] = laps_all['Position'] * first_lap

    return optimize_f1_analysis(laps_all, selected_GP, reg_clear_dd)


def optimize_f1_analysis(all_laps_df, selected_GP, reg_clear_dd):
    """
    Process laps and detect overtakes for logit modeling.
    """
    print("Preparing data and creating dummy variables...")

    # Save original columns
    all_laps_df['Year_original'] = all_laps_df['Year'].copy()
    all_laps_df['Driver_original'] = all_laps_df['Driver'].copy()

    # Create dummies
    dummy_columns = ["Driver", "Compound_Detail", "Year"]
    all_laps_df = pd.get_dummies(
        all_laps_df,
        columns=dummy_columns,
        prefix=dummy_columns,
        dtype=int
    )

    # Restore original columns
    all_laps_df['Year'] = all_laps_df['Year_original']
    all_laps_df['Driver'] = all_laps_df['Driver_original']
    all_laps_df = all_laps_df.drop(columns=['Year_original', 'Driver_original'])

    all_laps_df['LapNumber_square'] = all_laps_df['LapNumber'] ** 2

    if 'FirstLap_pos' not in all_laps_df.columns:
        first_lap_mask = (all_laps_df['LapNumber'] == 1.0)
        all_laps_df['FirstLap_pos'] = all_laps_df['Position'].where(first_lap_mask, 0)

    # TyreLife interactions
    compound_cols = [col for col in all_laps_df.columns if col.startswith("Compound_Detail_")]
    for col in compound_cols:
        compound_name = col.split("_", 2)[-1]
        all_laps_df[f"TyreLife_{compound_name}"] = all_laps_df["TyreLife"] * all_laps_df[col]

    # Predictions
    print("Calculating predictions...")
    relevant_data = all_laps_df[all_laps_df['GP'].isin(selected_GP)].copy()
    relevant_data['LapTime_pred'] = np.nan

    for gp in selected_GP:
        if gp not in reg_clear_dd:
            continue

        gp_mask = relevant_data['GP'] == gp
        gp_data = relevant_data[gp_mask]

        if gp_data.empty:
            continue

        try:
            fast_pred = reg_clear_dd[gp]
            data_list = gp_data.to_dict('records')
            predictions = fast_pred.predict_batch(data_list)
            relevant_data.loc[gp_mask, 'LapTime_pred'] = predictions
        except Exception as e:
            print(f"Error predicting for GP {gp}: {e}")
            continue

    # Prepare final dataset (use original laps_all without dummies for logit)
    print("Preparing final dataset...")

    # We need to reload clean data for the logit part
    laps_2021 = pd.read_csv(str(config.DATA_DIR / 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(str(config.DATA_DIR / 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(str(config.DATA_DIR / 'session_2023_V2.csv'))

    laps_all = pd.concat([laps_2021, laps_2022, laps_2023])
    laps_all = filter_laps(laps_all)
    laps_all['interval_front_real'] = laps_all['Interval_front'].copy()
    laps_all['DRS'] = (
        (laps_all['Interval_front'] < 1.0) &
        (laps_all['LapNumber'] > 2)
    ).astype(int)
    laps_all['Interval_front'] = np.exp(-laps_all['Interval_front'])
    laps_all['Interval_behind'] = np.exp(-laps_all['Interval_behind'])

    logit_df = laps_all[laps_all['GP'].isin(selected_GP)].copy()

    # Merge predictions
    pred_df = relevant_data[['Year', 'GP', 'Driver', 'LapNumber', 'LapTime_pred']].copy()
    logit_df = logit_df.merge(pred_df, on=['Year', 'GP', 'Driver', 'LapNumber'], how='left')
    logit_df = logit_df.sort_values(['Year', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)

    # Calculate last lap
    print("Calculating position metrics...")
    last_laps = logit_df.groupby(['Year', 'GP', 'Driver'])['LapNumber'].max().reset_index()
    last_laps.rename(columns={'LapNumber': 'LastLap'}, inplace=True)
    logit_df = logit_df.merge(last_laps, on=['Year', 'GP', 'Driver'])

    # Position changes
    logit_df['next_position'] = logit_df.groupby(['Year', 'GP', 'Driver'])['Position'].shift(-1)
    logit_df['position_gain/loss'] = logit_df['Position'] - logit_df['next_position']
    logit_df['position_gain/loss'] = logit_df['position_gain/loss'].fillna(0)

    # Detect overtakes
    print("Detecting overtakes...")
    logit_df['overtake'] = 0

    overtake_candidates = logit_df[
        (logit_df['LapNumber'] > 1) &
        (logit_df['position_gain/loss'] > 0) &
        (logit_df['TrackStatus'] == 1.0) &
        (logit_df['next_position'].notna())
    ].copy()

    if not overtake_candidates.empty:
        logit_indexed = logit_df.set_index(['Year', 'GP', 'LapNumber', 'Position'])

        for idx, row in overtake_candidates.iterrows():
            year, gp, lap = row['Year'], row['GP'], row['LapNumber']
            pos_act, next_pos = int(row['Position']), int(row['next_position'])

            try:
                rival_key = (year, gp, lap, pos_act - 1)
                if rival_key not in logit_indexed.index:
                    continue

                rival = logit_indexed.loc[rival_key]
                if isinstance(rival, pd.DataFrame):
                    rival = rival.iloc[0]

                if (rival['next_position'] >= next_pos and
                    rival['PitIn'] != 1.0 and
                    rival['PitOut'] != 1.0 and
                    rival['LastLap'] != lap):
                    logit_df.at[idx, 'overtake'] = 1
            except (KeyError, IndexError):
                continue

    # Calculate delta time
    print("Calculating time deltas...")
    logit_df['delta_t_lap'] = np.nan

    for (year, gp), group in logit_df.groupby(['Year', 'GP']):
        if group.empty:
            continue

        lap_times_dict = group.set_index(['Driver', 'LapNumber'])['LapTime_pred'].to_dict()
        position_dict = group.set_index(['LapNumber', 'Position'])['Driver'].to_dict()

        non_leaders = group[group['Position'] > 1].copy()

        for idx, row in non_leaders.iterrows():
            driver, lap, pos = row['Driver'], row['LapNumber'], int(row['Position'])

            rival_key = (lap, pos - 1)
            if rival_key not in position_dict:
                continue

            rival_driver = position_dict[rival_key]
            next_lap = lap + 1

            my_time = lap_times_dict.get((driver, next_lap))
            rival_time = lap_times_dict.get((rival_driver, next_lap))

            if pd.notna(my_time) and pd.notna(rival_time):
                logit_df.at[idx, 'delta_t_lap'] = my_time - rival_time

    print("Processing completed!")
    return logit_df


def train_overtake_logit(logit_df, gp: str, min_samples: int = 30,
                          min_positives: int = 5) -> Dict[str, Any]:
    """
    Train overtake logit model for a specific GP.

    Args:
        logit_df: Prepared DataFrame with overtake data
        gp: Grand Prix name
        min_samples: Minimum samples required
        min_positives: Minimum positive cases required

    Returns:
        Dictionary with model, metrics, and threshold
    """
    gp_df = logit_df[logit_df['GP'] == gp].copy()

    # Filter valid data
    gp_df = gp_df[gp_df['LapNumber'] > 3]
    gp_df = gp_df[gp_df['LapNumber'] < gp_df['LastLap']]
    gp_df = gp_df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake'])
    gp_df = gp_df[gp_df['interval_front_real'] <= 1.0]
    gp_df['delta_total'] = gp_df['interval_front_real'] + gp_df['delta_t_lap']

    print(f"\n{'='*60}")
    print(f"Training overtake model for: {gp}")
    print(f"{'='*60}")

    n_total = len(gp_df)
    n_overtakes = int(gp_df['overtake'].sum())
    overtake_rate = n_overtakes / n_total if n_total > 0 else 0

    print(f"Samples: {n_total} | Overtakes: {n_overtakes} | Rate: {overtake_rate:.1%}")

    if n_total < min_samples:
        raise ValueError(f"Insufficient samples: {n_total} < {min_samples}")

    if n_overtakes < min_positives:
        raise ValueError(f"Insufficient positive cases: {n_overtakes} < {min_positives}")

    # Prepare variables
    X = gp_df[['delta_total']].copy()
    y = gp_df['overtake'].copy()

    # Fit model
    X_sm = sm.add_constant(X)
    model = sm.Logit(y, X_sm).fit(
        disp=False,
        maxiter=1000,
        method='bfgs',
        warn_convergence=False
    )

    if not model.mle_retvals['converged']:
        print("Warning: Model did not converge")

    # Predictions
    y_pred_prob = model.predict(X_sm)

    # Adaptive threshold
    if n_overtakes > 0 and n_overtakes < len(y_pred_prob):
        sorted_probs = np.sort(y_pred_prob)[::-1]
        threshold = sorted_probs[n_overtakes - 1]
        threshold = max(threshold - 1e-10, 0.001)
    else:
        threshold = 0.5

    y_pred = (y_pred_prob >= threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    try:
        auc = roc_auc_score(y, y_pred_prob) if len(np.unique(y)) > 1 else np.nan
    except:
        auc = np.nan

    print(f"\nModel Summary:")
    print(model.summary())

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f} | AUC: {auc:.3f}")
    print(f"  Threshold: {threshold:.4f}")

    print(f"\nCoefficients:")
    for param, coef in model.params.items():
        pval = model.pvalues[param]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {param}: {coef:.4f} (p={pval:.4f}) {sig}")

    return {
        'model': model,
        'threshold': threshold,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'n_samples': n_total,
            'n_overtakes': n_overtakes
        }
    }


def train_missing_overtake_models(target_gps: list = None) -> Dict[str, Dict]:
    """
    Train overtake models for specified GPs.

    Args:
        target_gps: List of GP names to train

    Returns:
        Dictionary with GP name -> model info
    """
    if target_gps is None:
        target_gps = ['Emilia Romagna Grand Prix', 'Miami Grand Prix']

    # Load FastLinearPredictor for lap time predictions
    selected_GP = {gp: [] for gp in target_gps}
    selected_GP['Emilia Romagna Grand Prix'] = ['C3', 'C4', 'C5']
    selected_GP['Miami Grand Prix'] = ['C2', 'C3', 'C4']

    reg_clear_dd = {}
    models_dir = str(config.SIMULATION_DIR)

    for gp in target_gps:
        try:
            with open(f'{models_dir}/{gp}_fast_predictor.pkl', "rb") as f:
                reg_clear_dd[gp] = pickle.load(f)
            print(f"Loaded fast predictor for {gp}")
        except Exception as e:
            print(f"Error loading fast predictor for {gp}: {e}")
            return {}

    # Prepare data
    print("\nPreparing data for overtake analysis...")
    laps_2021, laps_2022, laps_2023, _ = load_and_prepare_data()
    logit_df = prepare_data_for_logit(laps_2021, laps_2022, laps_2023, selected_GP, reg_clear_dd)

    results = {}

    overtakes_dir = str(config.SIMULATION_DIR)

    for gp in target_gps:
        try:
            result = train_overtake_logit(logit_df, gp)

            # Save model
            filepath = f'{overtakes_dir}/{gp}_logit_V2.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"Saved model: {filepath}")

            # Save threshold
            filepath_threshold = f'{overtakes_dir}/{gp}_threshold.pkl'
            with open(filepath_threshold, 'wb') as f:
                pickle.dump(result['threshold'], f)
            print(f"Saved threshold: {filepath_threshold}")

            results[gp] = result

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
    results = train_missing_overtake_models()
