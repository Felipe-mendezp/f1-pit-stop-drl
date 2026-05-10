import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Evaluate RecurrentPPO models trained with 4 reward types (mix, time, position, points)
across 3 GPs (Bahrain, Miami, Dutch).

Stochastic variant -- fair comparison design:
  - Initial positions : fixed (2024 grid)
  - SC/VSC events     : pre-generated once per simulation using empirical probabilities,
                        then replayed identically for every reward type -> fair comparison
  - Rival strategies  : stochastic (pitstop + compound logit models)
  - All aux models    : overtake logit, rival logit, YF predictor all active

Generates a 3x3 grid boxplot: rows=GPs, columns=metrics (Times, Positions, Points).
Each subplot has 4 boxplots for the 4 reward types.

Usage:
    python scripts/evaluate_reward_comparison_stochastic.py
    python scripts/evaluate_reward_comparison_stochastic.py --n-sims 100
"""

import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Match the typography used in the paper figures.
mpl.rcParams.update({
    'font.family':           'sans-serif',
    'axes.titlesize':        17,
    'axes.labelsize':        18,
    'xtick.labelsize':       15,
    'ytick.labelsize':       15,
    'legend.fontsize':       16,
    'legend.title_fontsize': 16,
})

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from environment.f1_env_all_drivers import (
    create_f1_env_all_drivers, VSC_DURATION, SC_DURATION,
)
from evaluate_all_gps import (
    GP_CONFIGS, load_gp_config, NORM_OBS_KEYS,
)
from evaluate_reward_comparison import get_model_path
from model_loader import ModelLoader

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

N_SIMULATIONS = 500

# GP indices in GP_CONFIGS: Bahrain=0, Miami=2, Dutch=6
GP_INDICES = [0, 2, 6]

REWARD_TYPES  = ['mix', 'time', 'position', 'points']
REWARD_LABELS = ['mix', 'time', 'pos', 'points']
REWARD_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

METRIC_NAMES = ['Race Time [s]', 'Final Position', 'Championship Points']

# Consistent y-axis formatting across rows so all GPs share the same scale
# (mirrors F1_all_drivers/evaluate_reward_comparison_stochastic_oco_miami.py).
FINAL_POSITION_YTICKS = [3, 6, 9, 12, 15, 18]
FINAL_POSITION_YLIM   = (1.5, 20.5)
POINTS_YTICKS         = [0, 5, 10, 15, 20, 25]
POINTS_YLIM           = (0, 26)

# Cache file for simulation results
CACHE_FILE = os.path.join(str(config.RL_AGENTS_DIR), 'reward_comparison_stochastic_results.csv')


# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

@dataclass
class ExtendedSimResult:
    final_position: int
    total_time: float
    points: int
    changed_compound: bool


# -------------------------------------------------------------------------
# SC/VSC pre-generation
# -------------------------------------------------------------------------

def generate_sc_vsc_events(
    n_laps: int,
    race_vsc_prob: float,
    race_sc_prob: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Pre-generate SC/VSC active-lap lists for one race using the same stochastic
    logic as F1EnvAllDrivers._update_sc_vsc().

    Returns:
        laps_vsc: list of laps where VSC is active
        laps_sc:  list of laps where SC is active
    """
    rng = np.random.default_rng(seed)

    lap_vsc_prob = 1 - (1 - race_vsc_prob) ** (1 / n_laps) if race_vsc_prob > 0 else 0.0
    lap_sc_prob  = 1 - (1 - race_sc_prob)  ** (1 / n_laps) if race_sc_prob  > 0 else 0.0

    laps_vsc: List[int] = []
    laps_sc:  List[int] = []
    remaining_vsc = 0
    remaining_sc  = 0

    for lap in range(1, n_laps + 1):
        if remaining_vsc > 0:
            laps_vsc.append(lap)
            remaining_vsc -= 1
        elif remaining_sc > 0:
            laps_sc.append(lap)
            remaining_sc -= 1
        else:
            vsc_triggered = rng.random() < lap_vsc_prob
            sc_triggered  = rng.random() < lap_sc_prob

            # If both triggered, choose by relative race probability
            if vsc_triggered and sc_triggered and race_vsc_prob > 0 and race_sc_prob > 0:
                vsc_share = race_vsc_prob / (race_vsc_prob + race_sc_prob)
                if rng.random() < vsc_share:
                    sc_triggered = False
                else:
                    vsc_triggered = False

            if vsc_triggered:
                laps_vsc.append(lap)
                remaining_vsc = VSC_DURATION - 1
            elif sc_triggered:
                laps_sc.append(lap)
                remaining_sc = SC_DURATION - 1

    return laps_vsc, laps_sc


def pregenenerate_race_scenarios(
    n_sims: int,
    n_laps: int,
    race_vsc_prob: float,
    race_sc_prob: float,
    base_seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    Pre-generate SC/VSC events for all n_sims races.
    Uses seeds base_seed, base_seed+1, ..., base_seed+n_sims-1.

    Returns list of (laps_vsc, laps_sc) tuples, one per simulation.
    """
    scenarios = []
    for i in range(n_sims):
        laps_vsc, laps_sc = generate_sc_vsc_events(
            n_laps, race_vsc_prob, race_sc_prob, seed=base_seed + i,
        )
        scenarios.append((laps_vsc, laps_sc))
    return scenarios


# -------------------------------------------------------------------------
# Simulation
# -------------------------------------------------------------------------

def run_single_race(
    model,
    vec_env,
    raw_env,
    seed: int,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_positions: List[int],
    deterministic_policy: bool = False,
) -> ExtendedSimResult:
    """
    Run one race with:
      - pre-determined SC/VSC events (same across all reward types)
      - stochastic rival strategies (logit models, empty rival_actions)
      - fixed initial positions
    """
    # Fix SC/VSC events; rivals are stochastic (rival_actions={})
    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions={},
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)
    obs = vec_env.reset()

    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)
    done          = False
    terminal_info = {}

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start,
            deterministic=deterministic_policy,
        )
        episode_start = np.zeros((1,), dtype=bool)
        obs, _, done, info = vec_env.step(action)
        done = done[0]
        if done and 'final_position' in info[0]:
            terminal_info = info[0]

    changed_compound = terminal_info.get('has_changed_compound', False)
    raw_final_pos    = terminal_info.get('final_position', raw_env.agent_position)
    final_pos        = 20 if not changed_compound else raw_final_pos
    total_time       = terminal_info.get('total_time', 0.0)
    points           = F1_POINTS.get(final_pos, 0)

    return ExtendedSimResult(
        final_position=final_pos,
        total_time=total_time,
        points=points,
        changed_compound=changed_compound,
    )


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def evaluate_reward_type_stochastic(
    config_entry: dict,
    reward_type: str,
    loader: ModelLoader,
    scenarios: List[Tuple[List[int], List[int]]],
    initial_positions: List[int],
    base_seed: int = 42,
) -> List[ExtendedSimResult]:
    """
    Evaluate one reward type against the pre-generated SC/VSC scenarios.
    Rivals use logit models (stochastic). SC/VSC events are fixed per scenario.
    """
    model_path = get_model_path(reward_type, config_entry['model_dir'])

    if not os.path.exists(model_path):
        print(f"    WARNING: Model not found: {model_path}")
        return []

    print(f"    Model: {model_path}")

    raw_env = create_f1_env_all_drivers(
        gp=config_entry['gp_name'],
        driver=config_entry['agent_driver'],
        deterministic=False,
        yf_enabled=True,
        initial_positions=initial_positions,
        loader=loader,
        verbose=False,
    )

    vec_env = DummyVecEnv([lambda: raw_env])

    algo_dir     = os.path.dirname(model_path)
    vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats.pkl')

    if os.path.exists(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training  = False
            vec_env.norm_reward = False
        except (ValueError, AttributeError) as e:
            print(f"    WARNING: Failed to load VecNormalize ({e})")
            vec_env = VecNormalize(vec_env, norm_reward=False,
                                   norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
            vec_env.training = False
    else:
        print(f"    WARNING: No VecNormalize stats at {vecnorm_path}")
        vec_env = VecNormalize(vec_env, norm_reward=False,
                               norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
        vec_env.training = False

    model = RecurrentPPO.load(model_path, env=vec_env)

    results = []
    n_sims  = len(scenarios)
    for i, (laps_vsc, laps_sc) in enumerate(scenarios):
        result = run_single_race(
            model, vec_env, raw_env,
            seed=base_seed + i,
            laps_vsc=laps_vsc,
            laps_sc=laps_sc,
            initial_positions=initial_positions,
        )
        results.append(result)

        if (i + 1) % 100 == 0:
            positions = [r.final_position for r in results]
            print(f"      {i+1}/{n_sims} | Mean pos: {np.mean(positions):.2f}")

    vec_env.close()
    return results


# -------------------------------------------------------------------------
# Cache helpers
# -------------------------------------------------------------------------

def save_cache(
    all_data: Dict[int, Dict[str, List[ExtendedSimResult]]],
    path: str = CACHE_FILE,
) -> None:
    rows = []
    for gp_idx, gp_data in all_data.items():
        for reward_type, results in gp_data.items():
            for r in results:
                rows.append({
                    'gp_idx': gp_idx,
                    'reward_type': reward_type,
                    'final_position': r.final_position,
                    'total_time': r.total_time,
                    'points': r.points,
                    'changed_compound': r.changed_compound,
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCache saved: {path}")


def load_cache(path: str = CACHE_FILE) -> Dict[int, Dict[str, List[ExtendedSimResult]]]:
    df = pd.read_csv(path)
    all_data: Dict[int, Dict[str, List[ExtendedSimResult]]] = {}
    for gp_idx, gp_df in df.groupby('gp_idx'):
        gp_data: Dict[str, List[ExtendedSimResult]] = {}
        for reward_type, rt_df in gp_df.groupby('reward_type'):
            gp_data[reward_type] = [
                ExtendedSimResult(
                    final_position=int(row.final_position),
                    total_time=float(row.total_time),
                    points=int(row.points),
                    changed_compound=bool(row.changed_compound),
                )
                for row in rt_df.itertuples()
            ]
        all_data[int(gp_idx)] = gp_data
    return all_data


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_reward_comparison(
    all_data: Dict[int, Dict[str, List[ExtendedSimResult]]],
    save_dir: str,
):
    fig, axes = plt.subplots(3, 3, figsize=(20, 11), sharex=True)
    gp_display_names = [GP_CONFIGS[i]['display_name'] for i in GP_INDICES]
    metric_keys      = ['total_time', 'final_position', 'points']

    for row, gp_idx in enumerate(GP_INDICES):
        gp_data = all_data.get(gp_idx, {})

        for col, (metric_name, metric_key) in enumerate(zip(METRIC_NAMES, metric_keys)):
            ax         = axes[row][col]
            box_data   = []
            colors_used = []
            labels_used = []

            for rt_idx, reward_type in enumerate(REWARD_TYPES):
                if reward_type in gp_data and gp_data[reward_type]:
                    values = [getattr(r, metric_key) for r in gp_data[reward_type]]
                    box_data.append(values)
                    colors_used.append(REWARD_COLORS[rt_idx])
                    labels_used.append(REWARD_LABELS[rt_idx])

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    labels=labels_used,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red',
                                   markeredgecolor='red', markersize=6),
                    showfliers=True,
                    flierprops=dict(marker='o', markerfacecolor='none',
                                    markeredgecolor='gray', markersize=4,
                                    linestyle='none'),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                )
                for patch, color in zip(bp['boxes'], colors_used):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)

            if row == 0:
                ax.set_title(metric_name, fontweight='bold')
            if col == 0:
                ax.set_ylabel(gp_display_names[row])

            ax.grid(axis='y', alpha=0.3, linestyle='-')
            ax.tick_params(axis='x')
            ax.tick_params(axis='y')
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight('normal')
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight('normal')

            # Consistent y-axis formatting across all rows so panels match the
            # paper figures (Bahrain / Miami / Dutch share the same scale).
            from matplotlib.ticker import MaxNLocator
            if col == 0:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            elif col == 1:
                ax.set_ylim(*FINAL_POSITION_YLIM)
                ax.set_yticks(FINAL_POSITION_YTICKS)
            elif col == 2:
                ax.set_ylim(*POINTS_YLIM)
                ax.set_yticks(POINTS_YTICKS)
                ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='red', markeredgecolor='red',
               markersize=8, label='Mean'),
    ]
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=True)

    pdf_path = os.path.join(save_dir, 'reward_comparison_stochastic_boxplot.pdf')
    png_path = os.path.join(save_dir, 'reward_comparison_stochastic_boxplot.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stochastic reward comparison -- pre-generated SC/VSC, stochastic rivals',
    )
    parser.add_argument('--n-sims', type=int, default=N_SIMULATIONS,
                        help=f'Simulations per reward type (default: {N_SIMULATIONS})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all simulations even if cache exists')
    parser.add_argument('--plot-only', action='store_true',
                        help='Load cache and regenerate plots without running simulations')
    args = parser.parse_args()

    gp_names = [GP_CONFIGS[i]['display_name'] for i in GP_INDICES]

    print('=' * 70)
    print('REWARD COMPARISON EVALUATION -- STOCHASTIC (FAIR)')
    print('=' * 70)
    print(f'  GPs              : {gp_names}')
    print(f'  Reward types     : {REWARD_TYPES}')
    print(f'  Simulations      : {args.n_sims}')
    print(f'  Base seed        : {args.seed}')
    print(f'  Initial positions: fixed (2024 grid)')
    print(f'  SC/VSC           : pre-generated (same for all reward types)')
    print(f'  Rival strategies : stochastic (pitstop + compound logit models)')
    print('=' * 70)

    use_cache = os.path.exists(CACHE_FILE) and not args.force and not args.plot_only
    plot_only = args.plot_only

    if plot_only or use_cache:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: Cache file not found: {CACHE_FILE}")
            print("Run without --plot-only to generate simulations first.")
            sys.exit(1)
        print(f"\nLoading results from cache: {CACHE_FILE}")
        all_data = load_cache(CACHE_FILE)
        print(f"  Loaded {sum(len(v) for gp in all_data.values() for v in gp.values())} simulation results.")
    else:
        loader   = ModelLoader()
        all_data: Dict[int, Dict[str, List[ExtendedSimResult]]] = {}

        for gp_idx in GP_INDICES:
            entry      = GP_CONFIGS[gp_idx]
            gp_config  = load_gp_config(entry['gp_name'])
            initial_positions = gp_config['initial_positions_list_2024']

            # Load GP-level SC/VSC probabilities from ModelLoader
            gp_models      = loader.load(entry['gp_name'])
            race_vsc_prob  = gp_models.vsc_prob
            race_sc_prob   = gp_models.sc_prob
            n_laps         = gp_models.n_laps

            print(f"\n{'=' * 70}")
            print(f"GP: {entry['display_name']}  ({entry['agent_driver']})")
            print(f"  n_laps={n_laps}, SC_prob={race_sc_prob:.2f}, VSC_prob={race_vsc_prob:.2f}")
            print(f"{'=' * 70}")

            # -- Pre-generate SC/VSC scenarios ONCE for all reward types --
            print(f"  Pre-generating {args.n_sims} SC/VSC scenarios...", end=' ')
            scenarios = pregenenerate_race_scenarios(
                n_sims=args.n_sims,
                n_laps=n_laps,
                race_vsc_prob=race_vsc_prob,
                race_sc_prob=race_sc_prob,
                base_seed=args.seed,
            )
            sc_count  = sum(1 for _, ls in scenarios if ls)
            vsc_count = sum(1 for lv, _ in scenarios if lv)
            print(f"done. Races with SC: {sc_count}/{args.n_sims}, VSC: {vsc_count}/{args.n_sims}")

            gp_results: Dict[str, List[ExtendedSimResult]] = {}

            for reward_type in REWARD_TYPES:
                print(f"\n  Reward type: {reward_type}")
                results = evaluate_reward_type_stochastic(
                    config_entry=entry,
                    reward_type=reward_type,
                    loader=loader,
                    scenarios=scenarios,
                    initial_positions=initial_positions,
                    base_seed=args.seed,
                )
                if results:
                    gp_results[reward_type] = results
                    positions = [r.final_position for r in results]
                    times     = [r.total_time     for r in results]
                    pts       = [r.points          for r in results]
                    print(f"    Summary: Mean pos={np.mean(positions):.2f}, "
                          f"Mean time={np.mean(times):.1f}s, "
                          f"Mean pts={np.mean(pts):.2f}")

            all_data[gp_idx] = gp_results

        save_cache(all_data)

    save_dir = str(config.RL_AGENTS_DIR)
    plot_reward_comparison(all_data, save_dir)

    print('\n' + '=' * 70)
    print('EVALUATION COMPLETE')
    print('=' * 70)
