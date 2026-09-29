import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Per-circuit summary of the decision-support experiment: how the counterfactual
outcome of each candidate action evolves across laps.

Produces FOUR separate figures (one metric each), vs the current lap:
    <name>_evolution_position.{pdf,png}   Average final position (y inverted: up = better)
    <name>_evolution_points.{pdf,png}     Average points
    <name>_evolution_top10.{pdf,png}      P(final pos. <= 10)
    <name>_evolution_regret.{pdf,png}     PAIRED delta in avg points of each arm
                                          vs the DRL choice (zero line)
Each figure has one line per action (distinct markers) with a +-2 SE band, a
dashed vertical line at every lap where the agent ACTUALLY pits, a top strip
marking the DRL action per lap (plus the lap-0 starting tyre square), and its
legend below. Laps where the DRL action is beaten by another arm by more than
2 paired SE are ringed on the strip: without that band roughly half of the
visible line crossings are Monte Carlo noise, not real disagreement.

The regret figure is the paper's central claim in one panel: the rollouts share
the environment seed across arms, so the paired SE of the per-lap difference
(~0.15 pts at M=500) is the correct noise scale — the raw ±2 SE bands of the
other three figures overlap visually even where the paired test is conclusive.
"No curve rises above zero outside its band" IS the alignment statement.

Usage:
    python scripts/plot_decision_evolution.py --model-dir Belgian_NOR \
        --pdf-dir out/pdf/Belgian_NOR --png-dir out/png/Belgian_NOR --name belgian_nor
"""

import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ACTION_COLORS = {1: '#4d4d4d', 2: '#daa520', 3: '#d62728', 0: '#1f77b4'}
ACTION_MARKERS = {1: 's', 2: '^', 3: 'o', 0: 'D'}  # Hard, Medium, Soft, No pit
ACTION_ORDER = [1, 2, 3, 0]
PIT_LAP_COLOR = '#2ca02c'      # dashed vertical line at the agent's real stop
MISALIGN_EDGE = '#c00000'      # ring on the strip when the DRL arm is beaten


def display_label(lab, paper=False):
    """'No pit' is really 'do not pit on THIS lap and carry on with the policy',
    not a zero-stop race: name it so the figure cannot be misread. The
    '(follow policy)' suffix makes explicit that the blue line re-joins the
    agent's own strategy after the current lap — e.g. it still pits later.
    The manuscript variant drops the suffix (the caption explains it)."""
    if lab != 'No pit':
        return lab
    return 'Stay out' if paper else 'Stay out (follow policy)'


def _contiguous_runs(laps):
    """[29,30,31,32] -> [(29,32)] ; [2,3,7] -> [(2,3),(7,7)]."""
    runs = []
    for l in sorted(set(laps)):
        if runs and l == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], l)
        else:
            runs.append((l, l))
    return runs


def draw_yf_spans(ax, sc_laps, vsc_laps, label=True):
    """Shade the real safety-car (SC) and virtual safety-car (VSC) lap windows."""
    handles = []
    for laps, color, name in [(sc_laps, '#8e7cc3', 'Safety car'),
                              (vsc_laps, '#c9bde8', 'Virtual SC')]:
        first = True
        for s, e in _contiguous_runs(laps or []):
            ax.axvspan(s - 0.5, e + 0.5, color=color, alpha=0.35, lw=0, zorder=0)
            if first and label:
                handles.append(Patch(facecolor=color, alpha=0.5, label=name))
                first = False
    return handles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--pdf-dir', default=None)
    ap.add_argument('--png-dir', default=None)
    ap.add_argument('--name', default=None, help='basename for output files (default: model-dir lower)')
    ap.add_argument('--paper-style', action='store_true',
                    default=bool(os.environ.get('F1_DS_PAPER_STYLE')),
                    help='Manuscript variant (2026-09-11): ±2 SE band only (no '
                         '±1 SD), shorter legend labels, no vertical "agent pits" '
                         'text, "Starting compound:" label on the strip.')
    ap.add_argument('--drop-last-laps', type=int, default=0,
                    help='Drop the K latest decision laps (penultimate-lap pit is a '
                         'degenerate edge case whose pit outcomes are artefactually good)')
    ap.add_argument('--drop-first-laps', type=int, default=0,
                    help='Drop the K earliest decision laps. Pitting on laps 1-3 '
                         'rejoins the still-bunched field and is unrecoverable '
                         '(overtakes need a gap <= 1 s), so those arms collapse to '
                         '~0 points and squash the y axis. This is a REAL effect of '
                         'the simulator, not an artefact — dropping the laps is a '
                         'presentation choice and the caption must then state that '
                         'the sweep starts later.')
    args = ap.parse_args()

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'decision_support')
    pfx = args.model_dir.lower()
    name = args.name or pfx

    # Initial tyre = the lap-0 entry of the prefix trace recorded in the metas.
    # Deriving it from the run itself keeps the strip honest: the side-file
    # initial_compounds.json is written by a separate pass and goes stale as soon
    # as the agent set changes (it claimed Hard/C1 for a Dutch model that starts
    # on Soft/C3). The file is only a fallback for metas without 'history'.
    init = None
    init_path = os.path.join(results_dir, 'initial_compounds.json')
    if os.path.exists(init_path):
        init = json.load(open(init_path)).get(args.model_dir)

    metas = sorted(glob.glob(os.path.join(results_dir, f"{pfx}_lap*_meta.json")),
                   key=lambda p: int(p.split('_lap')[-1].split('_meta')[0]))
    if not metas:
        raise SystemExit(f"No data for {args.model_dir} in {results_dir}")

    laps, rec_by_lap = [], {}
    avg_pos = defaultdict(list)
    avg_pts = defaultdict(list)
    p_le10 = defaultdict(list)
    se_pos = defaultdict(list)
    se_pts = defaultdict(list)
    se_p10 = defaultdict(list)
    # SD of the outcome distribution (what the per-lap histograms show), kept
    # separate from the SE of the mean: at M=500 they differ by ~22x.
    sd_pos = defaultdict(list)
    sd_pts = defaultdict(list)
    # PAIRED per-lap regret of each arm vs the DRL choice: the rollouts share
    # the environment seed across arms, so mean/SE of the per-rollout
    # difference is the right estimator (the noise scale the misalignment
    # rings already use). Feeds the <name>_evolution_regret figure.
    reg_mean = defaultdict(list)
    reg_se = defaultdict(list)
    misaligned = []  # laps where the DRL arm is beaten by > 2 paired SE (points)
    argmax_laps = []  # laps where the DRL arm is the strict argmax (mean points)
    labels = None
    n_laps = None
    sc_laps, vsc_laps = [], []
    for mp in metas:
        n = int(mp.split('_lap')[-1].split('_meta')[0])
        meta = json.load(open(mp))
        labels = meta['action_labels']
        n_laps = meta.get('n_laps', n)
        if meta.get('history'):
            start = next((c for l, c, _p in meta['history'] if l == 0), None)
            if start is not None:
                comps = meta.get('available_compounds') or []
                act = comps.index(start) + 1 if start in comps else None
                if act is not None:
                    init = {'compound': start, 'action': act,
                            'label': labels.get(str(act), labels.get(act, f'C{start}'))}
        sc_laps = meta.get('laps_sc_2024') or sc_laps
        vsc_laps = meta.get('laps_vsc_2024') or vsc_laps
        # Fully-simulated variant: the yellow flags are whatever the common
        # prefix realized, recorded by the rollout script.
        if not sc_laps and not vsc_laps and meta.get('prefix_yf_laps'):
            sc_laps = [l for l, k in meta['prefix_yf_laps'] if k == 'SC']
            vsc_laps = [l for l, k in meta['prefix_yf_laps'] if k == 'VSC']
        csvp = os.path.join(results_dir, f"{pfx}_lap{n}.csv")
        pos = defaultdict(list); pts = defaultdict(list)
        for r in csv.DictReader(open(csvp)):
            a = int(r['action'])
            pos[a].append(int(r['final_position']))
            pts[a].append(float(r['points']))
        laps.append(n)
        rec = meta['recommended_action']
        rec_by_lap[n] = rec
        for a in ACTION_ORDER:
            vp = np.array(pos.get(a, [])); vt = np.array(pts.get(a, []))
            m = len(vp)
            avg_pos[a].append(np.mean(vp) if m else np.nan)
            avg_pts[a].append(np.mean(vt) if m else np.nan)
            p_le10[a].append(np.mean(vp <= 10) if m else np.nan)
            # SE of the mean; for the top-10 fraction, the binomial SE.
            se_pos[a].append(vp.std(ddof=1) / np.sqrt(m) if m > 1 else np.nan)
            se_pts[a].append(vt.std(ddof=1) / np.sqrt(m) if m > 1 else np.nan)
            sd_pos[a].append(vp.std(ddof=1) if m > 1 else np.nan)
            sd_pts[a].append(vt.std(ddof=1) if m > 1 else np.nan)
            if m > 1:
                p = float(np.mean(vp <= 10))
                se_p10[a].append(np.sqrt(max(p * (1 - p), 0.0) / m))
            else:
                se_p10[a].append(np.nan)
        # Paired regret of EVERY arm vs the recommendation (same rollout index
        # = same environment seed), for the regret figure.
        rec_v = np.array(pts.get(rec, []), dtype=float)
        for a in ACTION_ORDER:
            vt = np.array(pts.get(a, []), dtype=float)
            if len(vt) > 1 and len(vt) == len(rec_v):
                d = vt - rec_v
                reg_mean[a].append(d.mean())
                reg_se[a].append(d.std(ddof=1) / np.sqrt(len(d)))
            else:
                reg_mean[a].append(np.nan)
                reg_se[a].append(np.nan)
        # Paired test of the recommendation against the best alternative arm:
        # the rollouts share the environment seed across arms, so the paired SE
        # of the difference is the right noise scale (typically ~0.15 points,
        # i.e. much smaller than the raw per-arm SE).
        if rec in pts and len(pts[rec]) > 1:
            best_a = max(pts, key=lambda a: np.mean(pts[a]))
            if best_a == rec:
                argmax_laps.append(n)
            else:
                d = np.array(pts[best_a]) - rec_v
                se_d = d.std(ddof=1) / np.sqrt(len(d))
                if se_d > 0 and d.mean() > 2 * se_d:
                    misaligned.append(n)

    # Drop the K earliest decision laps (see --drop-first-laps).
    if args.drop_first_laps > 0 and len(laps) > args.drop_first_laps:
        k = args.drop_first_laps
        dropped = laps[:k]
        laps = laps[k:]
        for a in ACTION_ORDER:
            avg_pos[a] = avg_pos[a][k:]
            avg_pts[a] = avg_pts[a][k:]
            p_le10[a] = p_le10[a][k:]
            se_pos[a] = se_pos[a][k:]
            se_pts[a] = se_pts[a][k:]
            se_p10[a] = se_p10[a][k:]
            sd_pos[a] = sd_pos[a][k:]
            sd_pts[a] = sd_pts[a][k:]
            reg_mean[a] = reg_mean[a][k:]
            reg_se[a] = reg_se[a][k:]
        for n in dropped:
            rec_by_lap.pop(n, None)
        misaligned = [n for n in misaligned if n not in dropped]
        argmax_laps = [n for n in argmax_laps if n not in dropped]

    # Drop the K latest decision laps (artefactual final-lap pit outcomes).
    if args.drop_last_laps > 0 and len(laps) > args.drop_last_laps:
        keep = len(laps) - args.drop_last_laps
        dropped = laps[keep:]
        laps = laps[:keep]
        for a in ACTION_ORDER:
            avg_pos[a] = avg_pos[a][:keep]
            avg_pts[a] = avg_pts[a][:keep]
            p_le10[a] = p_le10[a][:keep]
            se_pos[a] = se_pos[a][:keep]
            se_pts[a] = se_pts[a][:keep]
            se_p10[a] = se_p10[a][:keep]
            sd_pos[a] = sd_pos[a][:keep]
            sd_pts[a] = sd_pts[a][:keep]
            reg_mean[a] = reg_mean[a][:keep]
            reg_se[a] = reg_se[a][:keep]
        for n in dropped:
            rec_by_lap.pop(n, None)
        misaligned = [n for n in misaligned if n not in dropped]
        argmax_laps = [n for n in argmax_laps if n not in dropped]

    laps = np.array(laps)
    # Laps where the agent ACTUALLY pits: its deterministic action there is a pit.
    pit_laps = [n for n in laps if rec_by_lap.get(n, 0) != 0]

    paper = args.paper_style

    def lab(a):
        return display_label(labels.get(str(a), labels.get(a, str(a))), paper)

    metrics = {
        'position': (avg_pos, se_pos, sd_pos, 'Average position', True),
        'points':   (avg_pts, se_pts, sd_pts, 'Average points', False),
        # No SD band for a fraction: its Bernoulli SD is sqrt(p(1-p)) <= 0.5 and
        # would swamp the panel without adding information.
        'top10':    (p_le10, se_p10, None, 'P(final pos. ≤ 10)', False),
        # Paired delta vs the DRL choice: the DRL choice is the zero line, the
        # band is +-2 PAIRED SE (same statistic as the misalignment rings).
        # No SD band: the paired difference has no meaningful outcome spread.
        'regret':   (reg_mean, reg_se, None,
                     'Δ avg points vs DRL choice (paired)', False),
    }

    saved = []
    for key, (data, sedata, sddata, ylabel, invert) in metrics.items():
        fig, (axs, ax) = plt.subplots(
            2, 1, figsize=(11, 4.6), sharex=True,
            gridspec_kw={'height_ratios': [0.35, 4]})

        # DRL action strip (+ lap-0 starting tyre)
        if init is not None:
            axs.scatter(0, 0, color=ACTION_COLORS.get(init['action'], 'k'),
                        s=70, marker='s', edgecolor='black', linewidth=1.2, zorder=5)
            axs.annotate(("Starting compound: " if paper else "start: ") + init['label'],
                         xy=(0, 0), xytext=(0, 0.6), ha='left', va='bottom',
                         fontsize=8.5)
        for n in laps:
            is_pit = rec_by_lap.get(n, 0) != 0
            # Positive marking: black edge where the DRL choice IS the strict
            # argmax of the four arms (complements the red misalignment ring —
            # "all edged, none ringed" reads as a perfect sweep at a glance).
            edge = 'black' if n in argmax_laps else 'none'
            axs.scatter(n, 0, color=ACTION_COLORS.get(rec_by_lap[n], 'k'),
                        s=70 if is_pit else 28, marker='s',
                        edgecolors=edge, linewidth=0.9,
                        zorder=5 if is_pit else 4)
        # Ring the laps where another arm beats the DRL one by > 2 paired SE.
        if misaligned:
            axs.scatter(misaligned, np.zeros(len(misaligned)), s=90, marker='o',
                        facecolors='none', edgecolors=MISALIGN_EDGE, linewidth=1.1,
                        zorder=6)
        axs.set_yticks([0]); axs.set_yticklabels(['DRL action'], fontsize=9)
        axs.set_ylim(-0.5, 0.5); axs.grid(False)
        for sp in ['top', 'right', 'left']:
            axs.spines[sp].set_visible(False)

        # Real safety-car / VSC windows (shaded), on both the strip and the axes.
        yf_handles = draw_yf_spans(ax, sc_laps, vsc_laps)
        draw_yf_spans(axs, sc_laps, vsc_laps, label=False)

        # The laps where the agent really stops (what the strip encodes with one
        # small square, easy to miss among 60), annotated with the compound it
        # takes so the "pit window opens -> agent pits" reading is immediate.
        for n in pit_laps:
            for a_ in (ax, axs):
                a_.axvline(n, color=PIT_LAP_COLOR, linestyle='--', linewidth=1.3,
                           alpha=0.9, zorder=1)
            if not paper:
                ax.annotate(f"agent pits → {lab(rec_by_lap[n])}",
                            xy=(n, 1.0), xycoords=('data', 'axes fraction'),
                            xytext=(3, -3), textcoords='offset points',
                            ha='left', va='top', rotation=90, fontsize=8,
                            color=PIT_LAP_COLOR, zorder=4)

        for a in ACTION_ORDER:
            mean = np.asarray(data[a], dtype=float)
            se = np.asarray(sedata[a], dtype=float)
            # TWO bands, because they answer different questions and conflating
            # them is confusing (they differ by sqrt(M) ~ 22x at M=500):
            #  - +-1 SD (faint, wide): the SPREAD of outcomes, i.e. what the
            #    per-lap histogram of plot_decision_support.py shows.
            #  - +-2 SE (darker, narrow): the PRECISION of the plotted mean,
            #    i.e. what the significance test / the ringed markers use.
            if sddata is not None and not paper:
                sd = np.asarray(sddata[a], dtype=float)
                ax.fill_between(laps, mean - sd, mean + sd,
                                color=ACTION_COLORS[a], alpha=0.10, lw=0, zorder=0)
            ax.fill_between(laps, mean - 2 * se, mean + 2 * se,
                            color=ACTION_COLORS[a], alpha=0.20, lw=0, zorder=1)
            ax.plot(laps, mean, linestyle='-', marker=ACTION_MARKERS[a], ms=4,
                    color=ACTION_COLORS[a], label=lab(a), zorder=3)

        if key == 'regret':
            # The DRL choice is the reference: anything ABOVE zero outside its
            # paired band would be a lap where an alternative beats the agent.
            ax.axhline(0.0, color='black', linewidth=1.2, zorder=2)
            ax.annotate('DRL choice (reference)', xy=(0.01, 0.97),
                        xycoords='axes fraction', ha='left', va='top',
                        fontsize=8.5, color='black')

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Current lap', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.5, (n_laps or laps.max()) + 0.5)
        if invert:
            ax.invert_yaxis()
        if key == 'top10':
            ax.set_ylim(-0.02, 1.02)
        # legend below the plot (action lines + bands + SC/VSC + agent's stops)
        handles, lbls = ax.get_legend_handles_labels()
        extra = list(yf_handles)
        extra.append(Patch(facecolor='0.4', alpha=0.30,
                           label=('±2 paired SE' if key == 'regret'
                                  else ('±2 SE' if paper
                                        else '±2 SE (precision of the mean)'))))
        if sddata is not None and not paper:
            extra.append(Patch(facecolor='0.4', alpha=0.10,
                               label='±1 SD (spread of outcomes)'))
        if pit_laps:
            extra.append(Line2D([], [], color=PIT_LAP_COLOR, linestyle='--',
                                linewidth=1.3, label='Agent pits here'))
        if misaligned:
            extra.append(Line2D([], [], color=MISALIGN_EDGE, marker='o', linestyle='none',
                                markerfacecolor='none', markersize=8,
                                label='DRL arm beaten (>2 SE)'))
        handles += extra; lbls += [h.get_label() for h in extra]
        ax.legend(handles, lbls, ncol=min(len(handles), 6 if paper else 5),
                  fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  frameon=True)

        fig.tight_layout()

        for d, fmt in [(args.pdf_dir, 'pdf'), (args.png_dir, 'png')]:
            if not d:
                continue
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{name}_evolution_{key}.{fmt}")
            fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
            saved.append(path)
        if not (args.pdf_dir or args.png_dir):
            path = os.path.join(results_dir, f"{name}_evolution_{key}.pdf")
            fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
            saved.append(path)
        plt.close(fig)

    print('Saved:\n  ' + '\n  '.join(saved))


if __name__ == '__main__':
    main()
