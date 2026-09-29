import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Alignment check for a decision-support run: per decision lap, compare the DRL
deterministic action against the action that maximizes P(final pos <= 10) in
the counterfactual rollouts. Reports the per-lap gap
    gap(n) = max_a P10(a) - P10(a_DRL)
and flags laps where the gap exceeds the Monte Carlo noise scale,
max(--thresh, 2 x the PAIRED SE of the difference).

Usage:
    python scripts/check_alignment.py --tag belgian_nor_masked --thresh 0.05
"""

import csv
import json
import glob
import argparse
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', required=True, help='file prefix, e.g. belgian_nor_masked')
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--thresh', type=float, default=0.05)
    args = ap.parse_args()

    rd = args.results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'decision_support')
    metas = sorted(glob.glob(os.path.join(rd, f"{args.tag}_lap*_meta.json")),
                   key=lambda p: int(p.split('_lap')[-1].split('_meta')[0]))
    if not metas:
        raise SystemExit(f"No data for tag {args.tag} in {rd}")

    flagged = []
    print(f"{'lap':>4} {'DRL':>10} {'best':>10} {'P10(DRL)':>9} {'P10(best)':>9} {'gap':>6}")
    for mp in metas:
        n = int(mp.split('_lap')[-1].split('_meta')[0])
        meta = json.load(open(mp))
        labels = {int(k): v for k, v in meta['action_labels'].items()}
        rec = meta['recommended_action']
        pos = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(rd, f"{args.tag}_lap{n}.csv"))):
            pos[int(r['action'])].append(int(r['final_position']))
        p10 = {a: np.mean(np.array(v) <= 10) for a, v in pos.items()}
        best = max(p10, key=p10.get)
        gap = p10[best] - p10[rec]
        # The arms share environment seeds per rollout index, so compare against
        # the PAIRED noise scale instead of a fixed threshold only.
        noise = args.thresh
        if best != rec and len(pos[best]) == len(pos[rec]) > 1:
            d = (np.array(pos[best]) <= 10).astype(float) - (np.array(pos[rec]) <= 10)
            se = d.std(ddof=1) / np.sqrt(len(d))
            noise = max(args.thresh, 2 * se)
        mark = '  <-- FLAG' if gap > noise else ''
        flagged.extend([n] if gap > noise else [])
        print(f"{n:>4} {labels[rec].split(' ')[0]:>10} {labels[best].split(' ')[0]:>10} "
              f"{p10[rec]:>9.3f} {p10[best]:>9.3f} {gap:>6.3f}{mark}")

    print(f"\nflagged laps (gap > max({args.thresh}, 2 paired SE)): {flagged or 'none'}")


if __name__ == '__main__':
    main()
