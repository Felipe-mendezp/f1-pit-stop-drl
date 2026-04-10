"""
Visualize agent stints for a given reward model across 3 GPs.
Shows 3 random races per GP to illustrate the agent's pit stop behavior.

Usage:
    python -m visualization.stint_visualizer              # default: time
    python -m visualization.stint_visualizer --reward mix
"""

import config

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from evaluate_all_gps_2024 import (
    GP_CONFIGS, build_rival_strategies, load_gp_config, NORM_OBS_KEYS,
)
from evaluate_reward_comparison import get_model_path, GP_INDICES
from model_loader import ModelLoader

# ─────────────────────────────────────────────────────────────────────────────
# Compound colors (F1 style: softest=red, hardest=white)
# ─────────────────────────────────────────────────────────────────────────────
COMPOUND_COLORS = {
    1: '#FFFFFF',   # C1 Hard - White
    2: '#F5F5F0',   # C2 Medium - Light
    3: '#F5A623',   # C3 Soft - Yellow/Orange
    4: '#7ED321',   # C4 - Green
    5: '#4A90D9',   # C5 - Blue
}

COMPOUND_EDGE = {
    1: '#888888',
    2: '#888888',
    3: '#CC8400',
    4: '#5A9E18',
    5: '#3570A8',
}

COMPOUND_CATEGORY = {
    1: 'Hard', 2: 'Medium', 3: 'Soft',
    4: 'Soft', 5: 'Soft',
}


def run_race_with_stints(model, vec_env, raw_env, seed, initial_compounds):
    """Run a single race and return stint data + final position + total time."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)
    obs = vec_env.reset()

    for rival in raw_env.rival_drivers:
        compound = initial_compounds.get(rival.name)
        if compound is not None:
            rival.compound = compound
            rival.initial_compound = compound

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    done = False

    # Stint tracking: record compound at each lap
    lap_compounds = []  # compound used at each racing lap (1..n_laps)
    terminal_info = {}
    n_laps = raw_env.n_laps

    while not done:
        # Save compound BEFORE step (after previous step processed it)
        compound_before = raw_env.agent_driver.compound
        lap_before = raw_env.lap_number

        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start,
            deterministic=False,
        )
        episode_start = np.zeros((1,), dtype=bool)

        obs, reward, done_arr, info = vec_env.step(action)
        done = done_arr[0]

        if done and 'final_position' in info[0]:
            terminal_info = info[0]

        # After step: record the compound for this lap
        # On lap 0 the agent selects starting compound; racing starts at lap 1
        if lap_before == 0:
            # Compound is set after lap 0 step
            lap_compounds.append(raw_env.agent_driver.compound)
        elif lap_before >= 1 and not done:
            lap_compounds.append(raw_env.agent_driver.compound)
        elif done:
            # Final lap: use compound before step (env may auto-reset)
            lap_compounds.append(compound_before)

    # Convert lap_compounds to stints
    stints = []
    if lap_compounds:
        current = lap_compounds[0]
        start = 1
        for i, c in enumerate(lap_compounds[1:], start=2):
            if c != current:
                stints.append((current, start, i - 1))
                current = c
                start = i
        stints.append((current, start, len(lap_compounds)))

    final_pos = terminal_info.get('final_position', 20)
    total_time = terminal_info.get('total_time', 0.0)
    changed = terminal_info.get('has_changed_compound', False)
    if not changed:
        final_pos = 20  # DQ

    return stints, final_pos, total_time, changed


def plot_stints_grid(all_data, save_dir, reward_label='time'):
    """
    Generate a 3-row x 3-col grid: rows=GPs, cols=races.
    Each cell shows the agent's stints as a horizontal bar.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 5), sharey=True)

    for row, (gp_name, races) in enumerate(all_data):
        for col, (stints, final_pos, total_time, changed, seed) in enumerate(races):
            ax = axes[row][col]
            n_laps = stints[-1][2] if stints else 57

            for compound, start, end in stints:
                width = end - start + 1
                color = COMPOUND_COLORS.get(compound, '#808080')
                edge = COMPOUND_EDGE.get(compound, '#555555')

                rect = Rectangle(
                    (start, 0.1), width, 0.8,
                    facecolor=color, edgecolor=edge, linewidth=1.5,
                )
                ax.add_patch(rect)

                # Label inside bar
                mid = start + width / 2
                laps_str = str(width)
                ax.text(mid, 0.5, f'C{compound} ({laps_str})',
                        ha='center', va='center', fontsize=8, fontweight='bold')

            ax.set_xlim(0, n_laps + 1)
            ax.set_ylim(0, 1)
            ax.set_yticks([])

            # Title: position and time
            status = 'DQ' if not changed else f'P{final_pos}'
            ax.set_title(f'{status} | {total_time:.0f}s | seed={seed}',
                         fontsize=9)

            ax.set_xlabel('Lap', fontsize=8)
            ax.tick_params(axis='x', labelsize=7)

        # Row label (GP name)
        axes[row][0].set_ylabel(gp_name, fontsize=11, fontweight='bold')

    # Legend
    legend_compounds = set()
    for _, races in all_data:
        for stints, *_ in races:
            for c, _, _ in stints:
                if c in COMPOUND_COLORS:
                    legend_compounds.add(c)

    legend_elements = [
        mpatches.Patch(facecolor=COMPOUND_COLORS[c], edgecolor=COMPOUND_EDGE[c],
                       linewidth=1.5, label=f'C{c}')
        for c in sorted(legend_compounds)
    ]

    fig.legend(handles=legend_elements, loc='lower center',
               ncol=len(legend_elements), fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Agent Stints -- '{reward_label}' Reward (3 random races per GP)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, f'{reward_label}_reward_stints.pdf')
    png_path = os.path.join(save_dir, f'{reward_label}_reward_stints.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward', type=str, default='time',
                        choices=['mix', 'time', 'position', 'points'],
                        help='Reward type to visualize')
    args = parser.parse_args()
    reward_type = args.reward

    loader = ModelLoader()
    base_dir = str(config.ROOT_DIR)

    # 3 random seeds for the 3 races
    rng = np.random.default_rng(123)
    race_seeds = rng.integers(0, 10000, size=3).tolist()

    all_data = []  # list of (gp_display_name, [race1, race2, race3])

    for gp_idx in GP_INDICES:
        entry = GP_CONFIGS[gp_idx]
        gp_display = entry['display_name']
        print(f"\n{'='*60}")
        print(f"GP: {gp_display}")
        print(f"{'='*60}")

        gp_config = load_gp_config(entry['gp_name'])
        rival_strategies, initial_compounds = build_rival_strategies(entry)
        initial_positions = gp_config['initial_positions_list_2024']
        laps_vsc = gp_config['laps_vsc_2024']
        laps_sc = gp_config['laps_sc_2024']

        model_path = get_model_path(base_dir, entry['model_dir'], reward_type)
        if not os.path.exists(model_path):
            print(f"  WARNING: Model not found: {model_path}")
            continue

        # Create env
        raw_env = create_f1_env_all_drivers(
            gp=entry['gp_name'],
            driver=entry['agent_driver'],
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

        vec_env = DummyVecEnv([lambda: raw_env])
        algo_dir = os.path.dirname(model_path)
        vecnorm_path = os.path.join(algo_dir, 'vecnormalize_stats.pkl')

        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            vec_env = VecNormalize(
                vec_env, norm_reward=False,
                norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0,
            )
            vec_env.training = False

        model = RecurrentPPO.load(model_path, env=vec_env)

        races = []
        for i, seed in enumerate(race_seeds):
            stints, pos, time, changed = run_race_with_stints(
                model, vec_env, raw_env, seed, initial_compounds,
            )
            status = 'DQ' if not changed else f'P{pos}'
            stint_str = ' -> '.join(f'C{c}({e-s+1})' for c, s, e in stints)
            print(f"  Race {i+1} (seed={seed}): {status}, {time:.0f}s | {stint_str}")
            races.append((stints, pos, time, changed, seed))

        vec_env.close()
        all_data.append((gp_display, races))

    # Plot
    plot_stints_grid(all_data, base_dir, reward_label=reward_type)
