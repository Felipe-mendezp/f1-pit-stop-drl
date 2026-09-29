"""
Figure H.1 (appendix): lap times of the 2024 Belgian GP with the laps flagged
as outliers by Algorithm 1 marked in a different color (per-driver thin lines,
blue = non-outlier, orange = outlier), replicating Code/laptimes_outliers.pdf.

Usage:
    python scripts/plot_fig_h1_outlier_laptimes.py [--gp "Belgian Grand Prix"]
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat; compat.setup_all()

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.regressions import all_laps, filtrar_outliers_iqr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp', default='Belgian Grand Prix')
    parser.add_argument('--output', default='figs/appendix/figure_H1.png')
    args = parser.parse_args()

    laps = all_laps(outliers=True)
    laps['GP'] = laps['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')

    # ALL laps of the GP (yellow-flag laps included): everything not present
    # in the final clear-track non-outlier set is marked as outlier, matching
    # the ~8.6% removal quoted in the paper (prefilters + Algorithm 1).
    gp_laps = laps[laps['GP'] == args.gp].copy()
    _, non_outliers = filtrar_outliers_iqr(laps, args.gp)

    key = ['Year', 'Driver', 'LapNumber']
    kept = set(map(tuple, non_outliers[key].itertuples(index=False)))
    gp_laps['is_outlier'] = [tuple(r) not in kept
                             for r in gp_laps[key].itertuples(index=False)]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    # thin connecting line per driver/year (race trace)
    for (_, _), d in gp_laps.groupby(['Year', 'Driver']):
        d = d.sort_values('LapNumber')
        ax.plot(d['LapNumber'], d['LapTime'], color='black', lw=0.4, zorder=1)
    ok = gp_laps[~gp_laps['is_outlier']]
    out = gp_laps[gp_laps['is_outlier']]
    ax.scatter(ok['LapNumber'], ok['LapTime'], s=14, color='tab:blue',
               edgecolors='black', linewidths=0.3, label='Non-outlier', zorder=2)
    ax.scatter(out['LapNumber'], out['LapTime'], s=14, color='tab:orange',
               edgecolors='black', linewidths=0.3, label='Outlier', zorder=3)
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Times [s]')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    fig.savefig(args.output.replace('.png', '.pdf'), bbox_inches='tight')
    pct = 100 * len(out) / len(gp_laps)
    print(f"Saved {args.output}  (outliers: {len(out)}/{len(gp_laps)} = {pct:.1f}%)")


if __name__ == '__main__':
    main()
