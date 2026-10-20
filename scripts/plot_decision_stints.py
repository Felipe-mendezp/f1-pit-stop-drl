import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Stint diagram for the decision-support section: one horizontal bar per candidate
action at the decision lap n. All bars share the common (deterministic) prefix up
to lap n; from lap n onward each bar shows the stint structure implied by forcing
that action and letting the trained Recurrent PPO policy (restricted to
rule-feasible actions) decide deterministically, with a common random-event seed
across the four bars.

Style follows the QMIP-comparison stint plot (Fig. 5): Hard=white, Medium=yellow,
Soft=red, black edges, 'C3 (17)' stint labels.

Usage:
    python scripts/plot_decision_stints.py --gp "Belgian Grand Prix" \
        --driver Driver_NOR --model-dir Belgian_NOR --lap 20 \
        --out-pdf ../Code/Figures/decision_support/pdf/Belgian_NOR/decision_stints_lap20.pdf \
        --out-png ../Code/Figures/decision_support/png/Belgian_NOR/decision_stints_lap20.png
"""

import copy
import csv as _csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
# 2026-10-11 opt-in: entorno con neumaticos opcion3 2021-24 (agente 2021-23), ver ds_rollouts_world2024_20261011.py
if os.environ.get('F1_DS_WORLD2024') == '1':
    import ds_rollouts_world2024_20261011  # noqa: F401  (parchea ModelLoader.load)
from evaluate_all_gps import (load_gp_config, build_rival_strategies, GP_CONFIGS,
                              ALGO_CLASSES, NORM_OBS_KEYS)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from decision_support_rollouts import (batch_obs, snapshot_env, restore_from,
                                       run_prefix_multi, action_labels,
                                       _rule_infeasible, switch_future_to_stochastic)

# Paper Fig. 5 palette: relative hardness -> color (bars carry black edges).
STINT_COLORS = ['white', 'gold', 'red']  # hardest, middle, softest
PANEL_ORDER = [1, 2, 3, 0]  # Hard, Medium, Soft, No pit (same as histogram panels)


def continuation_compounds(model, vec_env, cap, forced_action, seed, n_laps,
                           sim_future=False):
    """Force `forced_action` at the decision lap, then deterministic policy
    (rule-feasible masked) to the flag. Returns list of (lap, compound) for the
    agent from the decision lap to the end."""
    env = restore_from(cap['snap'])
    env.np_random = np.random.default_rng(seed)
    if sim_future:
        switch_future_to_stochastic(env)
    lstm = copy.deepcopy(cap['lstm'])
    ep = np.zeros((1,), dtype=bool)

    seq = []  # (lap_completed, compound_during_that_lap, pitted_this_lap)
    lap0 = env.lap_number
    obs, _, done, _, _ = env.step(int(forced_action))
    seq.append((lap0, int(env.agent_driver.compound), forced_action != 0))

    while not done:
        cur_lap = env.lap_number
        norm = vec_env.normalize_obs(batch_obs(obs))
        action, new_lstm = model.predict(norm, state=lstm, episode_start=ep,
                                         deterministic=True)
        act = int(np.asarray(action).squeeze())
        if _rule_infeasible(env, act):
            # Deterministic fallback: redirect to the policy-preferred different
            # compound (first legal compound action by hardness order).
            act = next(a for a in range(1, env._n_compounds + 1)
                       if env.available_compounds[a - 1] != env.agent_driver.compound)
        lstm = new_lstm
        obs, _, done, _, _ = env.step(act)
        seq.append((cur_lap, int(env.agent_driver.compound), act != 0))
    return seq


def to_stints(compound_by_lap):
    """[(lap, compound, pitted)] -> [(start_lap, end_lap_exclusive, compound)].
    A new stint starts on every pit stop, even if the compound is unchanged
    (a same-compound stop must remain visible in the diagram)."""
    stints = []
    for lap, comp, pitted in compound_by_lap:
        # FIX (2026-09-24): lap 0 is the starting-compound choice of the env,
        # not a racing lap; drawing it made a 62-lap race sum to 63. Racing
        # laps are 1..n_laps and a stint (s, e) covers laps s..e-1 inclusive.
        if lap == 0:
            continue
        if stints and stints[-1][2] == comp and not pitted:
            stints[-1] = (stints[-1][0], lap + 1, comp)
        else:
            stints.append((lap, lap + 1, comp))
    return stints


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gp', default='Belgian Grand Prix')
    ap.add_argument('--driver', default='Driver_NOR')
    ap.add_argument('--model-dir', default='Belgian_NOR')
    ap.add_argument('--lap', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1000,
                    help='Common event seed for the four continuations')
    ap.add_argument('--prefix-seed', type=int, default=0)
    ap.add_argument('--real-2024', action='store_true',
                    help='Condition the prefix and the four continuations on the '
                         'actual 2024 race (fixed SC/VSC and rivals); the four bars '
                         'then share the real event timeline exactly.')
    ap.add_argument('--real-yf', action='store_true',
                    help='Fix only the SC/VSC to the real race; rivals + lap times '
                         'stochastic. The four bars share a common seed so rivals '
                         'and events are identical across them (ceteris paribus).')
    ap.add_argument('--algo', default=os.environ.get('F1_DS_ALGO', 'RecurrentPPO'),
                    choices=list(ALGO_CLASSES.keys()))
    ap.add_argument('--out-pdf', default=None)
    ap.add_argument('--out-png', default=None)
    ap.add_argument('--results-dir', default=None,
                    help='Directory with the sweep CSV/meta this figure must '
                         'stay consistent with (default results/decision_support)')
    # Same opt-in env-physics fixes as decision_support_rollouts.py (2026-08-04):
    # the stint continuation must run under the SAME physics as the sweep CSVs.
    ap.add_argument('--pit-det-pass', action='store_true',
                    help='Fix (a): deterministic overtake resolution for '
                         'in-/out-lap drivers (no pit-lane weld).')
    ap.add_argument('--following-gap-scale', type=float, default=0.0,
                    help='Fix (b): denied overtakes settle at 0.4s + Exp(scale) '
                         'instead of the constant 0.4s. 0.0 = legacy.')
    ap.add_argument('--sim-future', action='store_true',
                    help='Protocolo 2026-08-13: continuaciones 100%% simuladas '
                         'desde la vuelta de decisión (debe coincidir con el '
                         'sweep que acompaña esta figura).')
    ap.add_argument('--paper-style', action='store_true',
                    default=bool(os.environ.get('F1_DS_PAPER_STYLE')),
                    help='Manuscript variant (2026-09-11): no title, no top/right '
                         'spines, "decision lap n" centred above the dashed line, '
                         'y label "Action in lap n", no green frame and no '
                         'outcome column (the histogram figure carries it).')
    args = ap.parse_args()
    paper = args.paper_style

    entry = next(e for e in GP_CONFIGS
                 if e['gp_name'] == args.gp and e['agent_driver'] == args.driver)
    gp_config = load_gp_config(args.gp)
    rival_actions, initial_compounds = build_rival_strategies(entry)
    loader = ModelLoader()
    raw_env = create_f1_env_all_drivers(
        gp=args.gp, driver=args.driver, deterministic=False, yf_enabled=True,
        initial_positions=gp_config['initial_positions_list_2024'],
        loader=loader, verbose=False)
    raw_env.pit_pass_deterministic = args.pit_det_pass
    raw_env.following_gap_scale = args.following_gap_scale
    if args.real_2024 or args.real_yf:
        raw_env.set_evaluation_mode(
            eval_mode=True, laps_vsc=gp_config['laps_vsc_2024'],
            laps_sc=gp_config['laps_sc_2024'],
            initial_positions=gp_config['initial_positions_list_2024'],
            rival_actions=(rival_actions if args.real_2024 else {}))
    vec_env = DummyVecEnv([lambda: raw_env])
    mp = os.path.join(str(config.RL_AGENTS_DIR), args.model_dir,
                      args.algo, 'best_model.zip')
    vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
    vec_env = VecNormalize.load(vn, vec_env) if os.path.exists(vn) else \
        VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS,
                     clip_reward=10000.0)
    vec_env.training = False; vec_env.norm_reward = False
    model = ALGO_CLASSES[args.algo].load(mp, env=vec_env)

    labels = action_labels(raw_env.available_compounds)
    comps = raw_env.available_compounds  # ascending C-number = hardest..softest
    color_of = {c: STINT_COLORS[i] for i, c in enumerate(comps)}
    n_laps = raw_env.n_laps

    # This figure must show the SAME race as the histogram of the same decision
    # lap, and the prefix (hence the whole stint structure) depends on the prefix
    # seed. Warn when a sweep for this tag exists under a different seed: with the
    # default 0 against a swept seed the two figures silently describe different
    # races (seen on Dutch: the stints put the stop on lap 31, the histogram on 37).
    suffix = '_real' if args.real_2024 else ('_realyf' if args.real_yf else '')
    _rd = args.results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'decision_support')
    _meta = os.path.join(_rd,
                         f'{args.model_dir.lower()}{suffix}_lap{args.lap}_meta.json')
    if os.path.exists(_meta):
        swept = json.load(open(_meta)).get('prefix_seed')
        if swept is not None and swept != args.prefix_seed:
            print(f"WARNING: the sweep for this lap used --prefix-seed {swept} but this "
                  f"figure is being drawn with {args.prefix_seed}; they will disagree.")

    # Common deterministic prefix to the decision lap; run_prefix_multi records
    # the agent's actual (lap, compound, pitted) history for laps 1..lap-1.
    caps = run_prefix_multi(model, vec_env, raw_env, [args.lap],
                            initial_compounds, args.prefix_seed)
    cap = caps[args.lap]
    pos0, comp0, life0 = cap['agent_state']
    prefix = cap['history']

    print(f"Decision lap {args.lap}: P{pos0}, on C{comp0}, tyre life {life0}. "
          f"DRL rec: {labels[cap['rec_action']]}")

    rows = []
    for a in PANEL_ORDER:
        seq = continuation_compounds(model, vec_env, cap, a, args.seed, n_laps,
                                     sim_future=args.sim_future)
        stints = to_stints(prefix + seq)
        # Every stop of the CONTINUATION (incl. the forced one) gets a glyph:
        # bar-boundary edges alone hide same-length neighbouring stints.
        cont_pits = [l for l, _c, p in seq if p]
        rows.append((a, stints, cont_pits))
        desc = ' | '.join(f"C{c} ({e - s})" for s, e, c in stints)
        print(f"  {labels[a]:14s}: {desc}")

    # Per-arm outcome column from the SAME sweep CSV that feeds the histogram
    # figure (mean points + paired delta vs the recommendation, +-2 paired SE):
    # ties Fig. 9 to Fig. 8 numerically instead of leaving the stint diagram
    # as a purely qualitative panel. Silently skipped if the sweep CSV is absent.
    outcome_txt = {}
    _csvp = _meta.replace('_meta.json', '.csv')
    if os.path.exists(_csvp) and os.path.exists(_meta):
        _pts = defaultdict(list)
        with open(_csvp) as f:
            for r in _csv.DictReader(f):
                _pts[int(r['action'])].append(float(r['points']))
        _rec = cap['rec_action']
        if _rec in _pts:
            _recv = np.array(_pts[_rec])
            for a, v in _pts.items():
                v = np.array(v)
                if a == _rec:
                    outcome_txt[a] = f"{v.mean():.2f} pts"
                elif len(v) == len(_recv) and len(v) > 1:
                    d = v - _recv
                    se2 = 2 * d.std(ddof=1) / np.sqrt(len(d))
                    outcome_txt[a] = (f"{v.mean():.2f} pts  "
                                      f"(Δ {d.mean():+.2f} ± {se2:.2f})")

    # Real SC/VSC windows to shade (meaningful whenever the events are pinned).
    sc_laps = gp_config['laps_sc_2024'] if (args.real_2024 or args.real_yf) else []
    vsc_laps = gp_config['laps_vsc_2024'] if (args.real_2024 or args.real_yf) else []

    def _runs(lps):
        out = []
        for l in sorted(set(lps or [])):
            if out and l == out[-1][1] + 1:
                out[-1] = (out[-1][0], l)
            else:
                out.append((l, l))
        return out

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(10, 3.2))
    yf_handles = []
    for lps, col, nm in [(sc_laps, '#8e7cc3', 'Safety car'),
                         (vsc_laps, '#c9bde8', 'Virtual SC')]:
        first = True
        for s, e in _runs(lps):
            ax.axvspan(s - 0.5, e + 0.5, color=col, alpha=0.35, lw=0, zorder=0)
            if first:
                yf_handles.append(Patch(facecolor=col, alpha=0.5, label=nm))
                first = False

    yticks, ylabels = [], []
    for i, (a, stints, cont_pits) in enumerate(rows):
        y = len(rows) - 1 - i
        yticks.append(y)
        name = labels[a] if a != 0 else 'Stay out'
        star = '  ★' if a == cap['rec_action'] else ''
        ylabels.append(name + star)
        for s, e, c in stints:
            # lap L occupies [L-0.5, L+0.5] so the bars line up with the ticks
            ax.barh(y, e - s, left=s - 0.5, height=0.62, color=color_of[c],
                    edgecolor='black', linewidth=1, zorder=2)
            if e - s >= 3:
                ax.text((s + e - 1) / 2, y, f"C{c} ({e - s})", ha='center', va='center',
                        fontsize=8.5, zorder=5)
        # One glyph per continuation stop (incl. the forced one): neighbouring
        # stints of the same colour are otherwise indistinguishable.
        for pl in cont_pits:
            ax.plot(pl - 0.5, y + 0.40, marker='v', color='black', ms=5,
                    clip_on=False, zorder=6)
        # Outcome column (right margin): mean points + paired delta vs ★, from
        # the same sweep CSV the histogram figure uses.
        if a in outcome_txt and not paper:
            is_rec = a == cap['rec_action']
            ax.annotate(outcome_txt[a], xy=(1.012, y),
                        xycoords=('axes fraction', 'data'),
                        ha='left', va='center', fontsize=8.5,
                        fontweight='bold' if is_rec else 'normal',
                        color='#2ca02c' if is_rec else 'black',
                        annotation_clip=False)
        if a == cap['rec_action'] and not paper:
            # Same convention as the histogram figure: green frame around the
            # recommended alternative.
            ax.add_patch(Rectangle((0.5, y - 0.40), n_laps, 0.80, fill=False,
                                   edgecolor='#2ca02c', linewidth=2.0, zorder=4))

    # Fade the common deterministic prefix: the four bars only differ from the
    # decision lap onward, and washing the shared history makes that immediate.
    # The decision at lap n is taken before lap n is run: boundary at n - 0.5.
    ax.axvspan(0.5, args.lap - 0.5, color='white', alpha=0.45, lw=0, zorder=3)

    ax.axvline(args.lap - 0.5, color='black', linestyle='--', linewidth=1.2, zorder=5)
    if paper:
        # Centred on the dashed line, just above the axes (where the title sat).
        ax.text(args.lap - 0.5, 1.02, f'decision lap {args.lap}',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=9.5)
    else:
        ax.text(args.lap - 0.5, len(rows) - 0.42, f' decision lap {args.lap}',
                ha='left', va='bottom', fontsize=9)
    if args.lap > 8:
        ax.text(args.lap / 2, len(rows) - 0.42, 'common history',
                ha='center', va='bottom', fontsize=8.5, color='0.35',
                style='italic')
    if outcome_txt and not paper:
        ax.annotate('avg pts (Δ vs ★, ±2 paired SE)', xy=(1.012, len(rows) - 0.35),
                    xycoords=('axes fraction', 'data'), ha='left', va='bottom',
                    fontsize=8, color='0.35', annotation_clip=False)

    if paper:
        # Only the bars and the x axis remain (2026-09-24: left spine and y tick
        # marks removed as well; the row labels stay).
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_ylabel(f'Action in lap {args.lap}', fontsize=11)
    else:
        ax.set_title(f"Stint structure implied by each action at lap {args.lap} — "
                     f"one representative rollout (common event seed {args.seed})",
                     fontsize=10)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel('Lap', fontsize=11)
    ax.set_xlim(0.5, n_laps + 0.5)
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.55)

    rel_names = ['Hard', 'Medium', 'Soft']
    handles = [Patch(facecolor=STINT_COLORS[i], edgecolor='black',
                     label=f'{rel_names[i]} (C{c})') for i, c in enumerate(comps)]
    handles += yf_handles
    ax.legend(handles=handles, ncol=3 + len(yf_handles), fontsize=9,
              loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=True)
    fig.tight_layout()

    saved = []
    for path, fmt in [(args.out_pdf, 'pdf'), (args.out_png, 'png')]:
        if not path:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        saved.append(path)
    if not saved:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'decision_support',
                           f"{args.model_dir.lower()}_stints_lap{args.lap}.pdf")
        fig.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
        saved.append(out)
    print('Saved: ' + ' , '.join(saved))


if __name__ == '__main__':
    main()
