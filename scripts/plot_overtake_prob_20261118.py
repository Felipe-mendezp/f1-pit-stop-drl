"""2026-11-18 - Probability of overtaking as a function of the predicted gap (supplementary Fig. L.1).

The manuscript draws the figure in LaTeX (pgfplots) from the coefficients of Table L.1. This script plots the same curves
from the same numbers, results/sim_v10eGQ_20261028/overtake_v10f/params.csv (const and slope of the logit of each circuit,
the pickles the simulator loads), and prints the values quoted in the text of Appendix L:
  - probability of overtaking at a predicted gap of zero (range over the eight circuits),
  - predicted gap at which the probability reaches 50 %,
  - predicted advantage (minus the gap) at which it reaches 95 %, the largest over the eight circuits.
A negative predicted gap means that the trailing car is predicted to finish the lap ahead.

    python scripts/plot_overtake_prob_20261118.py [--out figures_regenerated_v2.0.0/supplementary] [--xmax 3]
Output: <out>/overtake_probability.{pdf,png}
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, 'results', 'sim_v10eGQ_20261028', 'overtake_v10f', 'params.csv')
GPS = ['Bahrain', 'Belgian', 'Dutch', 'Hungarian', 'Miami', 'Saudi Arabian', 'Singapore', 'United States']
# Okabe-Ito palette, four line types and two widths (same styles as the manuscript's figure)
STYLES = [('black', '-', 1.5), ('#E69F00', '--', 1.5), ('#56B4E9', ':', 1.5), ('#009E73', '-.', 1.5),
          ('#0072B2', '-', 0.9), ('#D55E00', '--', 0.9), ('#CC79A7', ':', 0.9), ('0.5', '-.', 0.9)]


def prob(a, b, x):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures_regenerated_v2.0.0', 'supplementary'))
    ap.add_argument('--xmax', type=float, default=3.0)
    a = ap.parse_args()
    P = pd.read_csv(PARAMS).set_index('GP')
    P.index = [g.replace(' Grand Prix', '') for g in P.index]
    P = P.loc[GPS]
    p0 = prob(P.const, P.slope, 0.0).sort_values(ascending=False)       # legend order: decreasing probability at zero
    half = -P.const / P.slope                                           # predicted gap where the probability is 50 %
    gap95 = (np.log(0.95 / 0.05) - P.const) / P.slope                   # predicted gap where it reaches 95 % (negative)
    print('probability at a predicted gap of zero: %.1f %% (%s) to %.1f %% (%s)'
          % (100 * p0.iloc[-1], p0.index[-1], 100 * p0.iloc[0], p0.index[0]))
    print('gap for a 50 %% probability: %.2f s (%s) to %.2f s (%s)'
          % (half.max(), half.idxmax(), half.min(), half.idxmin()))
    print('predicted advantage needed for a 95 %% probability: at most %.2f s (%s)' % (-gap95.min(), gap95.idxmin()))
    os.makedirs(a.out, exist_ok=True)
    x = np.linspace(-a.xmax, a.xmax, 600)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for g, (c, ls, lw) in zip(p0.index, STYLES):
        ax.plot(x, prob(P.loc[g, 'const'], P.loc[g, 'slope'], x), color=c, ls=ls, lw=lw, label=g)
    ax.set_xlim(-a.xmax, a.xmax); ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted gap (s)'); ax.set_ylabel('Probability of overtaking')
    ax.grid(True, ls='--', color='0.85'); ax.legend(loc='upper right', fontsize=8, frameon=True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'overtake_probability.{ext}'), dpi=200)
    print('->', os.path.join(a.out, 'overtake_probability.pdf'))


if __name__ == '__main__':
    main()
