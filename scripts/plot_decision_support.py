import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Render the Decision Support Visualization figure (ANOR paper).

Reads the counterfactual-rollout outcomes produced by decision_support_rollouts.py
and draws a 2x2 grid of final-position histograms (one panel per candidate action).
Each panel is annotated with:
    - Average position (median for the recommended arm)
    - Average points
    - P(final position <= 10)
    - the PAIRED difference in mean points against the DRL-recommended arm, with
      a +-2 paired SE interval (the rollouts share environment seeds across arms;
      same statistic as the evolution figure and rank_alignment), so the reader
      can tell a real ranking from Monte Carlo noise — deltas whose interval
      includes 0 are explicitly labelled "statistical tie";
    - for non-recommended arms, the seed-by-seed head-to-head win rate of the
      recommendation (ties count 1/2).
A suptitle gives the decision context (GP, driver, lap, agent state, M) and a
one-line global verdict states whether any alternative significantly beats the
DRL choice. The DRL-recommended action is highlighted in green. If some other
arm is better beyond the paired interval it gets an orange frame instead of
hiding the disagreement. The x axis is cropped to the observed range of
finishing positions, with the points cutoff (P10/P11) marked.

Outputs both .pdf and .png (pass --out-pdf / --out-png).

Usage:
    python scripts/plot_decision_support.py --tag bahrain_alo_lap15 \
        --out-pdf out/pdf/x.pdf --out-png out/png/x.png
"""

import json
import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Panel display order and colors (repo convention: Medium=goldenrod, Soft=red).
PANEL_ORDER = [1, 2, 3, 0]  # Hard, Medium, Soft, No pit
ACTION_COLORS = {
    1: '#4d4d4d',   # Hard  -> dark grey
    2: '#daa520',   # Medium -> goldenrod
    3: '#d62728',   # Soft  -> red
    0: '#1f77b4',   # No pit -> blue
}
HIGHLIGHT_EDGE = '#2ca02c'  # green frame for the DRL recommendation
BEST_EDGE = '#e07b00'       # orange frame when another arm is significantly better


def display_label(lab, paper=False):
    """'No pit' means 'do not pit on THIS lap and carry on with the policy', not
    a zero-stop race: name it so the panel cannot be misread (same convention
    as plot_decision_evolution.py). The manuscript variant drops the suffix
    (the caption explains it)."""
    if lab != 'No pit':
        return lab
    return 'Stay out' if paper else 'Stay out (follow policy)'


def load_rows(csv_path):
    pos = defaultdict(list)
    pts = defaultdict(list)
    labels = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            a = int(r['action'])
            pos[a].append(int(r['final_position']))
            pts[a].append(float(r['points']))
            labels[a] = r['action_label']
    return ({a: np.array(v) for a, v in pos.items()},
            {a: np.array(v) for a, v in pts.items()}, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='bahrain_alo_lap15')
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--out-pdf', default=None, help='Output .pdf path')
    parser.add_argument('--out-png', default=None, help='Output .png path')
    parser.add_argument('--paper-style', action='store_true',
                        default=bool(os.environ.get('F1_DS_PAPER_STYLE')),
                        help='Manuscript variant (2026-09-11): no title/verdict '
                             '(the caption carries them), no frame on the '
                             'recommended arm, "-" instead of the star, shorter '
                             'text boxes, every 1..20 x tick and a tighter y '
                             'axis (auto-raised so no box covers a bar).')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Fixed upper limit of the y axis (2026-10-21); still raised if a stats box would cover a bar.')
    args = parser.parse_args()
    paper = args.paper_style

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'decision_support')

    csv_path = os.path.join(results_dir, f"{args.tag}.csv")
    meta_path = os.path.join(results_dir, f"{args.tag}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    pos_by, pts_by, labels = load_rows(csv_path)
    rec_action = meta['recommended_action']
    M = meta['M']
    lap = meta['lap']

    bins = np.arange(0.5, 21.5, 1.0)  # integer positions 1..20
    ymax = 0
    for v in pos_by.values():
        counts, _ = np.histogram(v, bins=bins)
        ymax = max(ymax, counts.max())
    ymax = ymax / M * (1.12 if paper else 1.30)  # fractions + headroom for annotations
    if args.ymax is not None:
        ymax = max(args.ymax, ymax / (1.12 if paper else 1.30))  # never below the tallest bar

    # Crop the x axis to the observed support: with the retrained agents every
    # arm finishes inside the points, so a fixed 1..20 axis squeezed all the mass
    # into the leftmost fifth of the panel and hid the differences.
    pos_max = max(int(v.max()) for v in pos_by.values())
    xhi = min(20.5, max(pos_max + 1, 8) + 0.5)
    if paper:
        xhi = 20.5  # 2026-10-21: every final position 1..20 on the axis
    xticks = (list(range(1, int(xhi) + 1)) if paper
              else [t for t in (1, 3, 5, 8, 10, 12, 15, 20) if t <= xhi])

    # Paired comparison against the recommended arm (arms share env seeds).
    # 2 * paired SE, NOT 1.96: same threshold as plot_decision_evolution.py and
    # rank_alignment.py, so a lap ringed there is framed orange here and
    # vice versa.
    rec_pts = pts_by.get(rec_action)
    rec_pos = pos_by.get(rec_action)
    best_action = max(pts_by, key=lambda a: float(np.mean(pts_by[a])))
    delta = {}
    winrate = {}  # P(recommendation finishes better | same seed), ties = 1/2
    for a, v in pts_by.items():
        if rec_pts is None or a == rec_action or len(v) != len(rec_pts):
            delta[a] = (0.0, 0.0)
            winrate[a] = None
            continue
        d = v - rec_pts
        delta[a] = (float(d.mean()), 2.0 * float(d.std(ddof=1)) / np.sqrt(len(d)))
        pa = pos_by[a]
        winrate[a] = float(np.mean(rec_pos < pa) + 0.5 * np.mean(rec_pos == pa))
    # "significantly better than the recommendation" = paired interval excludes 0.
    best_sig = (best_action != rec_action
                and delta[best_action][0] - delta[best_action][1] > 0)

    # 2026-10-21: shorter in the manuscript variant (author request)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2 if paper else 7.4), sharex=True, sharey=True)
    axes = axes.ravel()

    # --- Decision context + global verdict ---------------------------------
    drv = str(meta.get('driver', '')).replace('Driver_', '')
    st = meta.get('agent_state_at_lap') or {}
    comp = st.get('compound')
    comp_lab = next((display_label(l, paper) for l in labels.values()
                     if comp is not None and f'(C{comp})' in l),
                    f'C{comp}' if comp is not None else '?')
    ctx = (f"{meta.get('gp', '')} — {drv} — decision lap {lap}"
           f"  (P{st.get('position', '?')}, {comp_lab}, "
           f"{st.get('tyrelife', '?')} laps old)  —  M={M} rollouts per action")
    if best_action == rec_action:
        verdict = ("The DRL choice is the best arm in simulation — "
                   "no alternative beats it (paired 2·SE).")
        vcolor = HIGHLIGHT_EDGE
    elif not best_sig:
        dm, dci = delta[best_action]
        verdict = (f"No alternative significantly beats the DRL choice "
                   f"(best alternative +{dm:.2f} ± {dci:.2f} pts — statistical tie).")
        vcolor = HIGHLIGHT_EDGE
    else:
        dm, dci = delta[best_action]
        verdict = (f"⚠ {display_label(labels.get(best_action, str(best_action)), paper)} "
                   f"beats the DRL choice by {dm:+.2f} ± {dci:.2f} pts.")
        vcolor = BEST_EDGE
    if not paper:
        fig.suptitle(ctx, fontsize=11.5, y=0.995)
        fig.text(0.5, 0.952, verdict, ha='center', va='top', fontsize=10.5,
                 color=vcolor, fontweight='bold')

    panels = []  # (ax, stats-box artist, bar heights) for the overlap check
    for ax, a in zip(axes, PANEL_ORDER):
        v = pos_by.get(a)
        if v is None:
            ax.set_visible(False)
            continue
        heights, _, _ = ax.hist(v, bins=bins, weights=np.ones_like(v) / len(v),
                                color=ACTION_COLORS[a], alpha=0.85,
                                edgecolor='white', linewidth=0.5)

        # Points cutoff: P10 scores, P11 does not — makes the collapse of a
        # dominated arm (mass to the right of the line) immediately legible.
        if xhi > 10.5:
            ax.axvline(10.5, color='0.45', linestyle=':', linewidth=1.2, zorder=1)

        avg_pos = float(np.mean(v))
        avg_pts = float(np.mean(pts_by[a]))
        p_le10 = float(np.mean(v <= 10))
        median = float(np.median(v))

        title = display_label(labels.get(a, str(a)), paper)
        if a == rec_action:
            title += ' – DRL recommendation' if paper else '  ★ DRL recommendation'
            if not paper:
                for spine in ax.spines.values():
                    spine.set_edgecolor(HIGHLIGHT_EDGE)
                    spine.set_linewidth(2.4)
        elif a == best_action and best_sig:
            title += ' – best in simulation' if paper else '  ◆ best in simulation'
            for spine in ax.spines.values():
                spine.set_edgecolor(BEST_EDGE)
                spine.set_linewidth(2.0)
        ax.set_title(title, fontsize=12, fontweight='normal')

        if a == rec_action:
            lines = [f"Average position = {avg_pos:.2f}"]
            if not paper:
                lines.append(f"Median position = {median:.1f}")
            lines += [f"Average points = {avg_pts:.2f}",
                      f"P(final pos. ≤ 10) = {p_le10:.2f}"]
            if not paper:
                lines.append("(reference arm)")
        else:
            dm, dci = delta[a]
            dtxt = f"Δ points vs rec. = {dm:+.2f} ± {dci:.2f}"
            # An interval containing 0 is a TIE, not a disagreement: without
            # the label, a +0.00 ± 0.40 panel reads as an agent mistake.
            if abs(dm) <= dci and not paper:
                dtxt += " (statistical tie)"
            wr = winrate.get(a)
            lines = [f"Average position = {avg_pos:.2f}",
                     f"Average points = {avg_pts:.2f}",
                     f"P(final pos. ≤ 10) = {p_le10:.2f}"]
            if wr is not None:
                lines.append(f"DRL wins head-to-head = {wr:.0%}")
            lines.append(dtxt)
        box = ax.text(0.97, 0.95, "\n".join(lines),
                      transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                                edgecolor='0.7'))
        panels.append((ax, box, heights))

        ax.set_xlim(0.5, xhi)
        ax.set_ylim(0, ymax)
        ax.set_xticks(xticks)
        if paper:
            ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    for ax in axes[2:]:
        ax.set_xlabel('Final position', fontsize=11)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel('Fraction of simulations', fontsize=11)

    # Leave headroom for the suptitle + verdict lines.
    fig.tight_layout(rect=[0, 0, 1, 1] if paper else [0, 0, 1, 0.925])

    if paper:
        # The tighter y axis must never let a stats box cover a bar: raise
        # ymax until every bar under a box's x-span clears its lower edge.
        centers = 0.5 * (bins[1:] + bins[:-1])
        for _ in range(60):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            clash = False
            for ax, box, heights in panels:
                patch = box.get_bbox_patch()
                bb = (patch.get_window_extent(renderer) if patch is not None
                      else box.get_window_extent(renderer))
                (x0, y0), (x1, _) = ax.transData.inverted().transform(
                    [[bb.x0, bb.y0], [bb.x1, bb.y1]])
                under = (centers + 0.5 > x0) & (centers - 0.5 < x1)
                if np.any(heights[under] > y0):
                    clash = True
            if not clash:
                break
            ymax *= 1.05
            axes[0].set_ylim(0, ymax)  # sharey propagates to the other panels

    # 2026-10-21: manuscript fonts (opt-in F1_PAPER_FONTS=1, see paper_fonts_20261021.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paper_fonts_20261021; paper_fonts_20261021.apply(fig)
    saved = []
    for path, fmt in [(args.out_pdf, 'pdf'), (args.out_png, 'png')]:
        if not path:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        saved.append(path)
    if not saved:
        default = os.path.join(results_dir, 'decision_support_histograms.pdf')
        fig.savefig(default, format='pdf', dpi=300, bbox_inches='tight')
        saved.append(default)
    plt.close(fig)
    print('Saved: ' + ' , '.join(saved))


if __name__ == '__main__':
    main()
