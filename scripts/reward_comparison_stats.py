import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Pairwise statistical tests (Mann-Whitney U) for reward type comparison.

Loads simulation results from cache CSVs and produces:
  1. 3x3 grid of triangular p-value heatmaps (rows=GPs, cols=metrics).
  2. A CSV DataFrame with columns: gp, pair, metric, p_value.

Usage:
    python scripts/reward_comparison_stats.py
    python scripts/reward_comparison_stats.py --stochastic
"""

import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from evaluate_all_gps import GP_CONFIGS

# Match the typography used in the paper figures (see
# F1_all_drivers/reward_comparison_stats.py). Must come AFTER imports because
# evaluate_all_gps sets font.family at module scope.
mpl.rcParams.update({
    'font.family':           'sans-serif',
    'axes.titlesize':        17,
    'axes.labelsize':        18,
    'xtick.labelsize':       15,
    'ytick.labelsize':       15,
    'legend.fontsize':       16,
    'legend.title_fontsize': 16,
})

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

BASE_DIR   = str(config.RL_AGENTS_DIR)
# GP indices in GP_CONFIGS: Bahrain=0, Miami=2, Dutch=6
GP_INDICES = [0, 2, 6]

REWARD_TYPES  = ['mix', 'time', 'position', 'points']
REWARD_LABELS = ['mix', 'time', 'pos', 'points']

METRIC_COLS   = ['total_time', 'final_position', 'points']
METRIC_NAMES  = ['Race Time [s]', 'Final Position', 'Championship Points']

ALPHA = 0.05

CACHE_FILES = {
    'deterministic': os.path.join(BASE_DIR, 'reward_comparison_results.csv'),
    'stochastic':    os.path.join(BASE_DIR, 'reward_comparison_stochastic_results.csv'),
}
PLOT_FILES = {
    'deterministic': ('reward_comparison_stats.pdf',   'reward_comparison_stats.png'),
    'stochastic':    ('reward_comparison_stochastic_stats.pdf', 'reward_comparison_stochastic_stats.png'),
}
DF_FILES = {
    'deterministic': 'reward_comparison_pvalues.csv',
    'stochastic':    'reward_comparison_stochastic_pvalues.csv',
}


# -------------------------------------------------------------------------
# Statistical tests
# -------------------------------------------------------------------------

def compute_pairwise_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (gp_idx, metric, pair) combination compute the two-sided
    Mann-Whitney U p-value, then apply Holm-Bonferroni correction per panel
    (each GP x metric combination = 6 comparisons).

    Returns a DataFrame with columns:
        gp, pair, metric, p_value, p_value_adjusted
    """
    pairs = list(itertools.combinations(REWARD_TYPES, 2))
    pair_labels = [f"{a}-{b.replace('position', 'pos')}" for a, b in pairs]

    all_rows = []
    for gp_idx in GP_INDICES:
        gp_name = GP_CONFIGS[gp_idx]['display_name']
        gp_df   = df[df['gp_idx'] == gp_idx]

        for metric_col, metric_name in zip(METRIC_COLS, METRIC_NAMES):
            panel_rows = []
            for (rt_a, rt_b), label in zip(pairs, pair_labels):
                vals_a = gp_df.loc[gp_df['reward_type'] == rt_a, metric_col].values
                vals_b = gp_df.loc[gp_df['reward_type'] == rt_b, metric_col].values

                if len(vals_a) == 0 or len(vals_b) == 0:
                    p_value = float('nan')
                else:
                    _, p_value = mannwhitneyu(vals_a, vals_b, alternative='two-sided')

                panel_rows.append({
                    'gp':      gp_name,
                    'pair':    label,
                    'metric':  metric_name,
                    'p_value': p_value,
                })

            # Holm-Bonferroni adjustment across the 6 pairs in this panel
            raw_pvals = [r['p_value'] for r in panel_rows]
            _, adj_pvals, _, _ = multipletests(raw_pvals, alpha=ALPHA, method='holm')
            for r, adj in zip(panel_rows, adj_pvals):
                r['p_value_adjusted'] = adj

            all_rows.extend(panel_rows)

    return pd.DataFrame(all_rows)


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def build_pvalue_matrix(pval_df: pd.DataFrame, gp_name: str, metric_name: str) -> np.ndarray:
    """
    Build the Holm-adjusted p-value matrix for the given GP and metric,
    then return the 3x3 slice with the empty first column (mix) and last
    row (points) removed.

    Returned shape: (3, 3)
      rows -> mix, time, pos   (REWARD_LABELS[0:3])
      cols -> time, pos, points (REWARD_LABELS[1:4])
    Valid cells: i <= j (upper triangle including diagonal of the slice).
    """
    n = len(REWARD_LABELS)  # 4
    mat = np.full((n, n), np.nan)

    sub = pval_df[(pval_df['gp'] == gp_name) & (pval_df['metric'] == metric_name)]

    for i in range(n):
        for j in range(i + 1, n):
            rt_a = REWARD_TYPES[i]
            rt_b = REWARD_TYPES[j]
            label = f"{rt_a}-{rt_b.replace('position', 'pos')}"
            row = sub[sub['pair'] == label]
            if not row.empty:
                mat[i, j] = row.iloc[0]['p_value_adjusted']

    # Remove first column (mix, always empty) and last row (points, always empty)
    return mat[0:3, 1:4]


def plot_stats_grid(pval_df: pd.DataFrame, save_dir: str, variant: str) -> None:
    """3x3 grid of triangular p-value heatmaps: rows=GPs, cols=metrics."""
    gp_display_names = [GP_CONFIGS[i]['display_name'] for i in GP_INDICES]

    # Sliced matrix is 3x3: rows=mix/time/pos, cols=time/pos/points
    n = 3
    row_labels = REWARD_LABELS[0:3]  # ['mix', 'time', 'pos']
    col_labels = REWARD_LABELS[1:4]  # ['time', 'pos', 'points']

    fig, axes = plt.subplots(3, 3, figsize=(20, 11), sharex=True, sharey=True)

    for row, gp_name in enumerate(gp_display_names):
        for col, metric_name in enumerate(METRIC_NAMES):
            ax  = axes[row][col]
            mat = build_pvalue_matrix(pval_df, gp_name, metric_name)  # shape (3,3)

            # Valid cells: i <= j (upper triangle incl. diagonal of the slice)
            sig_mat = np.where(~np.isnan(mat), (mat < ALPHA).astype(float), np.nan)

            cmap = matplotlib.colors.ListedColormap(['#D9534F', '#5CB85C'])
            bounds = [-0.5, 0.5, 1.5]
            norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            ax.imshow(sig_mat, cmap=cmap, norm=norm, aspect='auto')

            # Annotate valid cells (i <= j in sliced coords)
            for i in range(n):
                for j in range(i, n):
                    val = mat[i, j]
                    if not np.isnan(val):
                        txt = f"{val:.3f}" if val >= 0.001 else "<.001"
                        ax.text(j, i, txt, ha='center', va='center',
                                fontsize=18, color='white', fontweight='bold')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight('normal')
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight('normal')

            # Hide invalid cells (strictly lower triangle: i > j)
            for i in range(n):
                for j in range(i):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        color='white', zorder=2,
                    ))

            if row == 0:
                ax.set_title(metric_name, fontweight='bold')
            if col == 0:
                ax.set_ylabel(gp_name)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5CB85C', label=f'p < {ALPHA}'),
        Patch(facecolor='#D9534F', label=f'p ≥ {ALPHA}'),
    ]
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=True)

    pdf_name, png_name = PLOT_FILES[variant]
    pdf_path = os.path.join(save_dir, pdf_name)
    png_path = os.path.join(save_dir, png_name)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pairwise Mann-Whitney U tests for reward comparison results',
    )
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic simulation cache instead of deterministic')
    args = parser.parse_args()

    variant   = 'stochastic' if args.stochastic else 'deterministic'
    cache_path = CACHE_FILES[variant]

    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found: {cache_path}")
        print("Run the corresponding evaluate script first.")
        sys.exit(1)

    print(f"Loading cache: {cache_path}")
    df = pd.read_csv(cache_path)
    print(f"  {len(df)} rows loaded.")

    print("\nComputing pairwise Mann-Whitney U tests...")
    pval_df = compute_pairwise_pvalues(df)
    print(f"  {len(pval_df)} comparisons computed.")

    # Save DataFrame
    df_path = os.path.join(BASE_DIR, DF_FILES[variant])
    pval_df.to_csv(df_path, index=False)
    print(f"\nResults saved: {df_path}")

    # Print summary table (adjusted p-values)
    print("\n" + "=" * 70)
    print(pval_df.pivot_table(
        index=['gp', 'metric'], columns='pair', values='p_value_adjusted',
    ).round(4).to_string())
    print("=" * 70)

    # Generate plots
    plot_stats_grid(pval_df, BASE_DIR, variant)

    print("\nDone.")
