"""
Reproduce Figure 3 of the paper:
Estimated tire performance with Equation (1) before (left panel) and after
(right panel) removing outliers, for the 2024 Belgian Grand Prix.

The y-axis indicates the additional lap time compared to a new hard-tire
compound; the x-axis represents tire usage (in laps).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

import argparse
import re

import numpy as np
import matplotlib.pyplot as plt

from models.regressions import all_laps, calibrar_regresion_gp, filtrar_outliers_iqr


# Mapping from compound code to display label
COMPOUND_LABELS = {
    'C0': 'Hard (C0)',
    'C1': 'Hard (C1)',
    'C2': 'Hard (C2)',
    'C3': 'Medium (C3)',
    'C4': 'Soft (C4)',
    'C5': 'Soft (C5)',
}

COMPOUND_COLOURS = {
    'soft':   '#E60023',
    'medium': '#F1B416',
    'hard':   '#111111',
}


def role_for_compound(compound: str, compounds_in_gp):
    """Return 'soft', 'medium' or 'hard' based on hardness ordering of GP compounds."""
    ordered = sorted(compounds_in_gp, key=lambda c: int(c[1:]))  # Cn -> n
    if compound == ordered[0]:
        return 'hard'
    if compound == ordered[-1]:
        return 'soft'
    return 'medium'


def extract_compound_coefficients(model, compounds):
    """
    Pull (intercept, slope) coefficients per compound from a fitted statsmodels OLS model.

    The reference compound has its own contribution absorbed into the intercept,
    so we re-anchor every compound to share the same baseline:

        additional_lap_time(c, j) = beta_{c,1} + beta_{c,2} * j

    where beta_{c,1} is set to 0 for the reference compound.
    """
    params = model.params

    # Figure out the reference compound from the parameter names
    pattern = re.compile(r"reference='([^']+)'")
    refs = [m.group(1) for name in params.index for m in [pattern.search(name)] if m]
    compound_ref = refs[0] if refs else None

    fixed = {}
    slope = {}
    for c in compounds:
        if c == compound_ref:
            fixed[c] = 0.0
        else:
            key = f"C(Compound_Detail, Treatment(reference='{compound_ref}'))[T.{c}]"
            fixed[c] = params.get(key, np.nan)
        slope[c] = params.get(f"TyreLife:C(Compound_Detail)[{c}]", np.nan)
    return fixed, slope


def plot_panel(ax, fixed, slope, compounds_in_gp, max_laps, title):
    """Plot one panel of additional lap time vs tire life for the listed compounds."""
    lap_grid = np.arange(0, max_laps + 1)
    # Use the hard compound at lap 0 as the baseline (0 additional time)
    hard = sorted(compounds_in_gp, key=lambda c: int(c[1:]))[0]
    base = fixed[hard] + slope[hard] * 0  # = fixed[hard]

    for c in compounds_in_gp:
        role = role_for_compound(c, compounds_in_gp)
        y = (fixed[c] + slope[c] * lap_grid) - base
        ax.plot(
            lap_grid,
            y,
            label=role.capitalize(),
            color=COMPOUND_COLOURS[role],
            lw=2,
        )
    ax.set_xlabel('Tire Usage [laps]')
    ax.set_ylabel('Additional Lap Time [s]')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_laps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp', default='Belgian Grand Prix',
                        help='Grand Prix name (default: Belgian Grand Prix)')
    parser.add_argument('--max-laps', type=int, default=44,
                        help='Maximum tire usage (laps) to display (default: 44)')
    parser.add_argument('--output', default='figs/figure_3_outlier_removal.png',
                        help='Output image path (PNG)')
    args = parser.parse_args()

    # Load the 2021-2023 historical data with the simple (median+15s) filter
    # disabled, so that outliers are still present. all_laps(outliers=True)
    # returns the dataset before the IQR procedure (Algorithm 1).
    laps_with_outliers = all_laps(outliers=True)

    # Apply the same dataset normalisations used by train_regressions.py
    laps_with_outliers['GP'] = laps_with_outliers['GP'].replace(
        'Styrian Grand Prix', 'Austrian Grand Prix'
    )

    if args.gp not in set(laps_with_outliers['GP'].unique()):
        raise SystemExit(f"GP '{args.gp}' not found in historical data")

    # ---- Left panel: regression on raw data (with outliers) ----
    laps_gp_raw = laps_with_outliers[
        (laps_with_outliers['GP'] == args.gp) &
        (laps_with_outliers['TrackStatus'] == 1.0)
    ].copy().reset_index(drop=True)
    model_raw, _ = calibrar_regresion_gp(laps_gp_raw)

    # ---- Right panel: regression after Algorithm 1 (IQR outlier removal) ----
    model_clean, _ = filtrar_outliers_iqr(laps_with_outliers, args.gp)

    # The compounds present in the GP (after filtering categorical encodings)
    compounds_in_gp = sorted(
        laps_gp_raw['Compound_Detail'].unique(),
        key=lambda c: int(c[1:]),
    )

    fixed_raw, slope_raw = extract_compound_coefficients(model_raw, compounds_in_gp)
    fixed_clean, slope_clean = extract_compound_coefficients(model_clean, compounds_in_gp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    plot_panel(axes[0], fixed_raw, slope_raw,
               compounds_in_gp, args.max_laps, 'With outliers')
    plot_panel(axes[1], fixed_clean, slope_clean,
               compounds_in_gp, args.max_laps, 'After outlier removal')
    axes[1].legend(title='Compound', loc='upper left')
    fig.suptitle(f'Tire Performance — {args.gp} (Eq. 1)', y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 3 to {args.output}")


if __name__ == '__main__':
    main()
