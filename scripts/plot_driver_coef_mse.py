"""
Reproduce Figure 4 of the paper:
MSE between estimated and ground-truth driver coefficients for each Grand
Prix of the 2024 season.

  - "With correction":   coefficients estimated via Equation (2) using
    recent race, qualifying, and free-practice data.
  - "Without correction": coefficients estimated using only historical data
    from previous years (i.e., taken from the most recent past GP for each
    driver).

For each 2024 GP we build per-driver features (Prev_race_1/2/3, Best FP,
Best Qualy) and fit the Equation (2) OLS on an EXPANDING WINDOW: only GPs
that occur EARLIER in the 2024 calendar than the target GP are used for
training (FIX 2026-07-21: the previous leave-one-GP-out scheme trained on
GPs *after* the target, a temporal leakage that inflated the "With
correction" curve). Early-season GPs without enough prior data get no
corrected estimate (shown as gaps). Use --loo to reproduce the old leaky
behaviour for comparison. The ground truth is the coefficient obtained by
training Equation (1) on lap times from the target GP itself, available in
`data/coef_drivers_2024.csv`.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from config import DATA_DIR


# 2024 calendar order (used to look up "previous 3 races" per driver).
CALENDAR_2024 = [
    'Bahrain Grand Prix',
    'Saudi Arabian Grand Prix',
    'Australian Grand Prix',
    'Japanese Grand Prix',
    'Chinese Grand Prix',
    'Miami Grand Prix',
    'Emilia Romagna Grand Prix',
    'Monaco Grand Prix',
    'Canadian Grand Prix',
    'Spanish Grand Prix',
    'Austrian Grand Prix',
    'British Grand Prix',
    'Hungarian Grand Prix',
    'Belgian Grand Prix',
    'Dutch Grand Prix',
    'Italian Grand Prix',
    'Azerbaijan Grand Prix',
    'Singapore Grand Prix',
    'United States Grand Prix',
    'Mexico City Grand Prix',
    'Sao Paulo Grand Prix',
    'Las Vegas Grand Prix',
    'Qatar Grand Prix',
    'Abu Dhabi Grand Prix',
]

# GPs displayed in Figure 4 (sprint weekends excluded, matching the paper).
DISPLAY_GPS = [
    'Bahrain', 'Saudi Arabian', 'Australian', 'Japanese',
    'Emilia Romagna', 'Monaco', 'Canadian', 'Spanish', 'British',
    'Hungarian', 'Belgian', 'Dutch', 'Italian', 'Azerbaijan',
    'Singapore', 'Mexico City', 'Las Vegas', 'Abu Dhabi',
]


def short_name(full_name: str) -> str:
    """'Bahrain Grand Prix' -> 'Bahrain'."""
    return full_name.replace(' Grand Prix', '').strip()


def build_feature_table(coefs_2024: pd.DataFrame, practice_2024: pd.DataFrame) -> pd.DataFrame:
    """
    For every (Driver, GP) row in `coefs_2024`, attach:
        Prev_race_1, Prev_race_2, Prev_race_3  - driver's coef at the 1, 2, 3
            most recent past GPs in the 2024 calendar (NaN if not available);
        BestFP, BestQualy - taken from `practice_2024`.

    Returns a DataFrame with one row per (Driver, GP) and the features above
    plus the target column `Coef` (the ground-truth coefficient).
    """
    cal_index = {gp: i for i, gp in enumerate(CALENDAR_2024)}
    coefs_2024 = coefs_2024[coefs_2024['GP'].isin(cal_index)].copy()
    coefs_2024['cal_idx'] = coefs_2024['GP'].map(cal_index)

    rows = []
    for (driver, gp), grp in coefs_2024.groupby(['Driver', 'GP']):
        target_idx = cal_index[gp]
        history = (
            coefs_2024[(coefs_2024['Driver'] == driver) &
                       (coefs_2024['cal_idx'] < target_idx)]
            .sort_values('cal_idx', ascending=False)
            ['Coef']
            .tolist()
        )
        prev_1 = history[0] if len(history) >= 1 else np.nan
        prev_2 = history[1] if len(history) >= 2 else np.nan
        prev_3 = history[2] if len(history) >= 3 else np.nan

        rows.append({
            'Driver': driver,
            'GP': gp,
            'cal_idx': target_idx,
            'Prev_race_1': prev_1,
            'Prev_race_2': prev_2,
            'Prev_race_3': prev_3,
            'Coef': grp['Coef'].iloc[0],
        })
    feats = pd.DataFrame(rows)

    # Attach best free-practice and qualifying lap times from practice_2024
    fp = practice_2024.copy()
    fp['BestFP'] = fp[['BestLapTime_FP1', 'BestLapTime_FP2', 'BestLapTime_FP3']].min(axis=1)
    fp = fp[['Driver', 'GP', 'BestFP', 'BestLapTime_Qualy']].rename(
        columns={'BestLapTime_Qualy': 'BestQualy'}
    )
    feats = feats.merge(fp, on=['Driver', 'GP'], how='left')
    return feats


def fit_correction(train: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS for Equation (2)/(A.1) on a training subset (rows with all features)."""
    cols = ['Prev_race_1', 'Prev_race_2', 'Prev_race_3', 'BestFP', 'BestQualy']
    train_clean = train.dropna(subset=cols + ['Coef'])
    X = sm.add_constant(train_clean[cols])
    y = train_clean['Coef']
    return sm.OLS(y, X).fit(), cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='figs/figure_4_driver_coef_mse.png',
                        help='Output image path (PNG)')
    parser.add_argument('--loo', action='store_true',
                        help='Use the old leave-one-GP-out training scheme '
                             '(TEMPORAL LEAKAGE: trains on GPs after the '
                             'target; only for comparison)')
    parser.add_argument('--min-train-rows', type=int, default=30,
                        help='Minimum complete training rows required to fit '
                             'the correction for a GP (default: 30)')
    args = parser.parse_args()

    coefs_2024 = pd.read_csv(os.path.join(DATA_DIR, 'coef_drivers_2024.csv'))
    practice_2024 = pd.read_csv(os.path.join(DATA_DIR, 'practice_2024.csv'))
    feats = build_feature_table(coefs_2024, practice_2024)

    mse_with, mse_without = [], []
    for short in DISPLAY_GPS:
        full = next((g for g in CALENDAR_2024 if short_name(g) == short), None)
        if full is None:
            mse_with.append(np.nan)
            mse_without.append(np.nan)
            continue

        target = feats[feats['GP'] == full]
        if args.loo:
            # Old scheme, kept only for comparison: trains on ALL other GPs,
            # including those AFTER the target (temporal leakage).
            train = feats[feats['GP'] != full]
        else:
            # Expanding window: only information available BEFORE the target
            # race, as Equation (2) requires.
            target_idx = feats.loc[feats['GP'] == full, 'cal_idx'].iloc[0] \
                if not target.empty else None
            train = feats[feats['cal_idx'] < target_idx] if target_idx is not None else feats.iloc[0:0]
        if target.empty:
            mse_with.append(np.nan)
            mse_without.append(np.nan)
            continue

        # ----- Without correction -------------------------------------
        # Predict the current GP coefficient using the driver's most recent
        # past coefficient (Prev_race_1).
        valid_no = target.dropna(subset=['Prev_race_1', 'Coef'])
        if valid_no.empty:
            mse_without.append(np.nan)
        else:
            err = valid_no['Coef'] - valid_no['Prev_race_1']
            mse_without.append(float((err ** 2).mean()))

        # ----- With correction ----------------------------------------
        try:
            cols_needed = ['Prev_race_1', 'Prev_race_2', 'Prev_race_3',
                           'BestFP', 'BestQualy', 'Coef']
            n_complete = len(train.dropna(subset=cols_needed))
            if n_complete < args.min_train_rows:
                print(f"[info] {short}: only {n_complete} complete training "
                      f"rows before this GP (<{args.min_train_rows}); "
                      f"no corrected estimate.")
                mse_with.append(np.nan)
                continue
            model, cols = fit_correction(train)
            valid_yes = target.dropna(subset=cols + ['Coef'])
            if valid_yes.empty:
                mse_with.append(np.nan)
            else:
                X_te = sm.add_constant(valid_yes[cols], has_constant='add')
                # Align columns (statsmodels can drop const if collinear)
                X_te = X_te.reindex(columns=model.params.index, fill_value=0.0)
                pred = model.predict(X_te)
                err = valid_yes['Coef'].values - pred.values
                mse_with.append(float((err ** 2).mean()))
        except Exception as e:
            print(f"[warn] fit failed for {short}: {e}")
            mse_with.append(np.nan)

    # ----- Plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(DISPLAY_GPS))
    ax.plot(x, mse_with, marker='^', ms=7, color='#1f77b4',
            lw=1.5, label='With correction')
    ax.plot(x, mse_without, marker='s', ms=6, color='#d62728',
            lw=1.5, ls='--', label='Without correction')
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_GPS, rotation=45, ha='right')
    ax.set_ylabel('Driver coefficient MSE')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 4 to {args.output}")

    # Also print the numerical results
    summary = pd.DataFrame({
        'GP': DISPLAY_GPS,
        'With correction': mse_with,
        'Without correction': mse_without,
    })
    print('\n' + summary.to_string(index=False))


if __name__ == '__main__':
    main()
