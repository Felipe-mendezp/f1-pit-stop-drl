"""2026-11-19 - Spread of the mean final position across the 10 training seeds (supplementary Fig. on seed selection).

One panel per circuit; one dot per training seed and algorithm, bar = median over the seeds. The values are the
validation simulations of the seed selection (200 races per seed, results/seed_selection_v5_20261010/v10f_v10g), not the
500 simulations of Figures 6-7 and Table P.1.

    python scripts/plot_seed_spread_20261119.py [--out figures_regenerated_v2.0.0/supplementary]
Output: <out>/seed_spread.{pdf,png}
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'results', 'seed_selection_v5_20261010', 'v10f_v10g',
                   'seed_summary_DQN_A2C_TRPO_PPO_RecurrentPPO_r578_dsq0.csv')
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']
LABELS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RPPO']
COLORS = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00']   # Okabe-Ito


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures_regenerated_v2.0.0', 'supplementary'))
    a = ap.parse_args()
    d = pd.read_csv(CSV)
    gps = list(dict.fromkeys(d.model_dir))
    rng = np.random.default_rng(0)                        # horizontal jitter only
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.7), sharex=True)
    for ax, g in zip(axs.ravel(), gps):
        for k, al in enumerate(ALGOS):
            s = d[(d.model_dir == g) & (d.algo == al)]
            ax.scatter(k + rng.uniform(-.12, .12, len(s)), s.pos_media, s=9, color=COLORS[k], alpha=.7, lw=0)
            ax.hlines(s.pos_media.median(), k - .3, k + .3, color='k', lw=1.4)
        ax.set_title(g.rsplit('_', 1)[0].replace('_', ' ') + ' GP', fontweight='bold', fontsize=7)
        ax.set_xticks(range(5)); ax.set_xticklabels(LABELS, fontsize=6.5)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))   # whole positions only
        ax.grid(axis='y', ls='--', color='0.88')
    for r in range(2):
        axs[r, 0].set_ylabel('Mean final position', fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.3)
    fig.subplots_adjust(wspace=0.22)
    fig.text(0.5, 0.015, 'Each dot is one of the 10 training seeds; the bar is the median.', ha='center', fontsize=7)
    os.makedirs(a.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'seed_spread.{ext}'), dpi=200)
    print('->', os.path.join(a.out, 'seed_spread.pdf'))


if __name__ == '__main__':
    main()
