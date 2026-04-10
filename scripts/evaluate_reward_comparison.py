import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Evaluate RecurrentPPO models trained with 4 reward types (mix, time, position, points)
across 3 GPs (Bahrain, Emilia Romagna, Dutch).

Generates a 3x3 grid boxplot: rows=GPs, columns=metrics (Times, Positions, Points).
Each subplot has 4 boxplots for the 4 reward types.

Usage:
    python scripts/evaluate_reward_comparison.py
    python scripts/evaluate_reward_comparison.py --n-sims 100
"""

import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List
from dataclasses import dataclass

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from evaluate_all_gps import (
    GP_CONFIGS, build_rival_strategies, load_gp_config, NORM_OBS_KEYS,
)
from model_loader import ModelLoader

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

N_SIMULATIONS = 500

# GP indices in GP_CONFIGS: Bahrain=0, Emilia Romagna=3, Dutch=6
GP_INDICES = [0, 3, 6]

# Reward types and display labels
REWARD_TYPES = ['mix', 'time', 'position', 'points']
REWARD_LABELS = ['mix', 'time', 'pos', 'points']
REWARD_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']  # blue, orange, green, pink

# Column metrics
METRIC_NAMES = ['Race Time [s]', 'Final Position', 'Points']

# Cache file for simulation results
CACHE_FILE = os.path.join(str(config.RL_AGENTS_DIR), 'reward_comparison_results.csv')


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
# Simulation
# -------------------------------------------------------------------------

def run_single_race_extended(
    model,
    vec_env,
    raw_env,
    seed: int,
    initial_compounds: Dict[str, int],
    deterministic_policy: bool = False,
) -> ExtendedSimResult:
    """Run a single race and capture position, total_time, and points."""
    # Seed all RNGs for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)
    obs = vec_env.reset()

    # Set each rival's starting compound
    for rival in raw_env.rival_drivers:
        compound = initial_compounds.get(rival.name)
        if compound is not None:
            rival.compound = compound
            rival.initial_compound = compound

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    done = False
    current_compound = None
    terminal_info = {}

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start,
            deterministic=deterministic_policy,
        )
        episode_start = np.zeros((1,), dtype=bool)

        is_lap_zero = (raw_env.lap_number == 0)
        agent_compound = raw_env.agent_driver.compound

        obs, reward, done, info = vec_env.step(action)
        done = done[0]

        # Track compound changes
        if is_lap_zero:
            current_compound = raw_env.agent_driver.compound
        elif current_compound is None:
            current_compound = agent_compound
        elif agent_compound != current_compound:
            current_compound = agent_compound

        if done and 'final_position' in info[0]:
            terminal_info = info[0]

    changed_compound = terminal_info.get('has_changed_compound', False)
    raw_final_pos = terminal_info.get('final_position', raw_env.agent_position)
    final_pos = 20 if not changed_compound else raw_final_pos
    total_time = terminal_info.get('total_time', 0.0)
    points = F1_POINTS.get(final_pos, 0)

    return ExtendedSimResult(
        final_position=final_pos,
        total_time=total_time,
        points=points,
        changed_compound=changed_compound,
    )


# -------------------------------------------------------------------------
# Model path resolution
# -------------------------------------------------------------------------

def get_model_path(reward_type: str, model_dir: str) -> str:
    """Return model path for given reward type."""
    if reward_type == 'mix':
        return os.path.join(
            str(config.RL_AGENTS_DIR), model_dir, 'RecurrentPPO', 'best_model.zip',
        )
    return os.path.join(
        str(config.RL_REWARD_DIR), model_dir,
        f'RecurrentPPO_{reward_type}', 'best_model.zip',
    )


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def evaluate_reward_type(
    config_entry: dict,
    reward_type: str,
    loader: ModelLoader,
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 42,
) -> List[ExtendedSimResult]:
    """Evaluate a single reward type for a given GP."""
    gp_config = load_gp_config(config_entry['gp_name'])
    rival_strategies, initial_compounds = build_rival_strategies(config_entry)

    initial_positions = gp_config['initial_positions_list_2024']
    laps_vsc = gp_config['laps_vsc_2024']
    laps_sc = gp_config['laps_sc_2024']

    model_path = get_model_path(reward_type, config_entry['model_dir'])

    if not os.path.exists(model_path):
        print(f"    WARNING: Model not found: {model_path}")
        return []

    print(f"    Model: {model_path}")

    # Create raw environment
    raw_env = create_f1_env_all_drivers(
        gp=config_entry['gp_name'],
        driver=config_entry['agent_driver'],
        deterministic=False,
        yf_enabled=True,
        initial_positions=initial_positions,
        loader=loader,
        verbose=False,
    )
    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions=rival_strategies,
    )

    # Wrap in DummyVecEnv + VecNormalize
    vec_env = DummyVecEnv([lambda: raw_env])

    algo_dir = os.path.dirname(model_path)
    vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats.pkl')

    if os.path.exists(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        except (ValueError, AttributeError) as e:
            print(f"    WARNING: Failed to load VecNormalize ({e})")
            vec_env = VecNormalize(
                vec_env, norm_reward=False,
                norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0,
            )
            vec_env.training = False
    else:
        print(f"    WARNING: No VecNormalize stats at {vecnorm_path}")
        vec_env = VecNormalize(
            vec_env, norm_reward=False,
            norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0,
        )
        vec_env.training = False

    model = RecurrentPPO.load(model_path, env=vec_env)

    results = []
    for i in range(n_simulations):
        result = run_single_race_extended(
            model, vec_env, raw_env,
            seed=base_seed + i,
            initial_compounds=initial_compounds,
        )
        results.append(result)

        if (i + 1) % 100 == 0:
            positions = [r.final_position for r in results]
            print(f"      {i+1}/{n_simulations} | Mean pos: {np.mean(positions):.2f}")

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
    """Generate 3x3 grid boxplot: rows=GPs, columns=metrics (Times, Positions, Points)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    gp_display_names = [GP_CONFIGS[i]['display_name'] for i in GP_INDICES]
    metric_keys = ['total_time', 'final_position', 'points']

    for row, gp_idx in enumerate(GP_INDICES):
        gp_data = all_data.get(gp_idx, {})

        for col, (metric_name, metric_key) in enumerate(
            zip(METRIC_NAMES, metric_keys)
        ):
            ax = axes[row][col]

            box_data = []
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
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markeredgecolor='red', markersize=6,
                    ),
                    showfliers=True,
                    flierprops=dict(
                        marker='o', markerfacecolor='none',
                        markeredgecolor='gray', markersize=4,
                        linestyle='none',
                    ),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                )

                for patch, color in zip(bp['boxes'], colors_used):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)

            # Column titles on top row only
            if row == 0:
                ax.set_title(metric_name, fontsize=14, fontweight='bold')

            # GP name as y-label on leftmost column only
            if col == 0:
                ax.set_ylabel(gp_display_names[row], fontsize=12, fontweight='bold')

            ax.grid(axis='y', alpha=0.3, linestyle='-')
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

            # Integer y-axis for positions column
            if col == 1:
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # Invert y-axis for points column (more points = lower = better)
            if col == 2:
                ax.invert_yaxis()

    # Legend at bottom center: red diamond = Mean
    legend_elements = [
        Line2D(
            [0], [0], marker='D', color='w',
            markerfacecolor='red', markeredgecolor='red',
            markersize=8, label='Mean',
        ),
    ]

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, wspace=0.25)

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=1,
        fontsize=12,
        frameon=True,
    )

    pdf_path = os.path.join(save_dir, 'reward_comparison_boxplot.pdf')
    png_path = os.path.join(save_dir, 'reward_comparison_boxplot.png')
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
        description='Evaluate reward type comparison across GPs',
    )
    parser.add_argument(
        '--n-sims', type=int, default=N_SIMULATIONS,
        help=f'Number of simulations per reward type (default: {N_SIMULATIONS})',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed (default: 42)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run all simulations even if cache exists',
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Load cache and regenerate plots without running simulations',
    )
    args = parser.parse_args()

    gp_names = [GP_CONFIGS[i]['display_name'] for i in GP_INDICES]

    print('=' * 70)
    print('REWARD COMPARISON EVALUATION')
    print('=' * 70)
    print(f'  GPs: {gp_names}')
    print(f'  Reward types: {REWARD_TYPES}')
    print(f'  Simulations per reward type: {args.n_sims}')
    print(f'  Base seed: {args.seed}')
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
        loader = ModelLoader()
        all_data: Dict[int, Dict[str, List[ExtendedSimResult]]] = {}

        for gp_idx in GP_INDICES:
            entry = GP_CONFIGS[gp_idx]

            print(f"\n{'=' * 70}")
            print(f"GP: {entry['display_name']}  ({entry['agent_driver']})")
            print(f"{'=' * 70}")

            gp_results: Dict[str, List[ExtendedSimResult]] = {}

            for reward_type in REWARD_TYPES:
                print(f"\n  Reward type: {reward_type}")
                results = evaluate_reward_type(
                    entry, reward_type, loader,
                    n_simulations=args.n_sims, base_seed=args.seed,
                )
                if results:
                    gp_results[reward_type] = results

                    positions = [r.final_position for r in results]
                    times = [r.total_time for r in results]
                    pts = [r.points for r in results]
                    print(f"    Summary: Mean pos={np.mean(positions):.2f}, "
                          f"Mean time={np.mean(times):.1f}s, "
                          f"Mean pts={np.mean(pts):.2f}")

            all_data[gp_idx] = gp_results

        save_cache(all_data)

    # Generate plot
    save_dir = str(config.RL_AGENTS_DIR)
    plot_reward_comparison(all_data, save_dir)

    print('\n' + '=' * 70)
    print('EVALUATION COMPLETE')
    print('=' * 70)
