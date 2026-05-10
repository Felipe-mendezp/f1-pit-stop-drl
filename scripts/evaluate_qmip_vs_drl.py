import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Evaluate QMIP optimization vs DRL models under realistic stochastic conditions.

Key differences from evaluate_all_gps.py:
  - SC/VSC events    : random (per-GP probabilities), NOT fixed from 2024
  - Rival strategies : stochastic (logit models), NOT fixed real strategies
  - Models compared  : 5 DRL algorithms + QMIP
  - SC/VSC scenarios are pre-generated once and replayed for all models (fair comparison)

Usage:
    python scripts/evaluate_qmip_vs_drl.py
    python scripts/evaluate_qmip_vs_drl.py --n-sims 5
    python scripts/evaluate_qmip_vs_drl.py --plot-only
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional
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

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO, TRPO

from environment.f1_env_all_drivers import (
    create_f1_env_all_drivers, VSC_DURATION, SC_DURATION,
)
from evaluate_all_gps import (
    GP_CONFIGS, load_gp_config, NORM_OBS_KEYS,
)
from evaluate_reward_comparison_stochastic import (
    generate_sc_vsc_events, pregenenerate_race_scenarios,
)
from model_loader import ModelLoader

# QMIP imports (in src/optimization/ — `src/` is on sys.path via config.py)
from optimization.qmip import solve_MIP_GUROBI, solve_MIP_GUROBI_start, Tires

# Simplified calculator for extracting betas
from optimization.utils_qmip import extract_simplified_calculator


# =============================================================================
# CONSTANTS
# =============================================================================

N_SIMULATIONS = 500

ALGO_ORDER = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']

ALGO_COLORS = {
    'DQN':          '#0072B2',  # blue       (Wong 2011)
    'A2C':          '#E69F00',  # orange
    'TRPO':         '#009E73',  # green
    'PPO':          '#CC79A7',  # pink
    'RecurrentPPO': '#56B4E9',  # sky blue
    'QMIP':         '#D55E00',  # vermilion
}

ALGO_ABBREV = {
    'DQN': 'DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'RPPO',
    'QMIP': 'QMIP',
}

ALGO_CLASSES = {
    'RecurrentPPO': RecurrentPPO,
    'PPO': PPO,
    'DQN': DQN,
    'A2C': A2C,
    'TRPO': TRPO,
}

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

CACHE_FILE = os.path.join(str(config.RL_AGENTS_DIR), 'qmip_vs_drl_stochastic_results.csv')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimResult:
    final_position: int
    total_reward: float
    pit_stops: int
    changed_compound: bool


# =============================================================================
# QMIP PARAMETER EXTRACTION
# =============================================================================

def build_qmip_params(loader: ModelLoader, gp_name: str) -> dict:
    """Extract QMIP parameters from the GP's lap time regression model.

    Returns dict with: betas_dict, time_stop, compound_to_qmip, qmip_to_compound,
                       available_compounds (sorted int list).
    """
    models = loader.load(gp_name)
    available_compounds_int = sorted(
        int(c[1:]) if isinstance(c, str) else c
        for c in models.compounds
    )

    calc = extract_simplified_calculator(models.laptime_clear, available_compounds_int)

    # Mapping: lowest C = Hard, middle = Medium, highest = Soft
    compound_to_qmip = {
        available_compounds_int[0]: "Hard",
        available_compounds_int[1]: "Medium",
        available_compounds_int[2]: "Soft",
    }
    qmip_to_compound = {v: k for k, v in compound_to_qmip.items()}

    betas_dict = {}
    for c_num, qmip_name in compound_to_qmip.items():
        beta_0 = calc.intercept + calc.compound_effects[c_num]
        beta_1 = calc.tyre_degradation[c_num]
        betas_dict[qmip_name] = [beta_0, beta_1]

    time_stop = calc.pit_in_cost + calc.pit_out_cost

    return {
        'betas_dict': betas_dict,
        'time_stop': time_stop,
        'compound_to_qmip': compound_to_qmip,
        'qmip_to_compound': qmip_to_compound,
        'available_compounds': available_compounds_int,
    }


# =============================================================================
# QMIP DECISION LOGIC
# =============================================================================

def qmip_choose_start_compound(
    betas_dict: dict,
    time_stop: float,
    n_laps: int,
    qmip_to_compound: dict,
    available_compounds: List[int],
    p: int = 3,
) -> int:
    """Call QMIP to choose starting compound. Returns env action (1-3)."""
    _, _, _, t0_opt = solve_MIP_GUROBI_start(
        betas_dict, time_stop, n=n_laps, w_current=0,
        t_previous=[], p=p, gamma=0, N=n_laps, verbose=False,
    )
    compound_num = qmip_to_compound[t0_opt]
    action = available_compounds.index(compound_num) + 1
    return action, t0_opt


def qmip_choose_action(
    betas_dict: dict,
    time_stop: float,
    n_remaining: int,
    t_current_name: str,
    w_current: float,
    t_previous: List[str],
    n_total: int,
    gamma: int,
    qmip_to_compound: dict,
    available_compounds: List[int],
    p: int = 3,
) -> int:
    """Call QMIP to decide pit/no-pit for current lap. Returns env action (0-3)."""
    if n_remaining <= 0:
        return 0

    try:
        x_values, y_values, _ = solve_MIP_GUROBI(
            betas_dict, time_stop, n=n_remaining,
            t_current=t_current_name, w_current=w_current,
            t_previous=t_previous, p=p, gamma=gamma, N=n_total,
            verbose=False,
        )
    except Exception:
        return 0  # Default: no pit on solver failure

    # If x[0] >= 1, stay on current tires
    if x_values[0] >= 1:
        return 0

    # x[0] == 0: pit now. Find which compound from y indicators.
    t_s = [1, 2, 3] * p
    for i, y_val in enumerate(y_values):
        if y_val > 0.5:
            tire_name = Tires[t_s[i] - 1]  # "Soft", "Medium", or "Hard"
            compound_num = qmip_to_compound[tire_name]
            action = available_compounds.index(compound_num) + 1
            return action

    # Fallback: no pit
    return 0


# =============================================================================
# QMIP RACE RUNNER
# =============================================================================

def run_single_race_qmip(
    raw_env,
    seed: int,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_positions: List[int],
    qmip_params: dict,
    p: int = 3,
) -> SimResult:
    """Run a single race with QMIP as the decision maker (raw Gymnasium env)."""
    betas_dict = qmip_params['betas_dict']
    time_stop = qmip_params['time_stop']
    compound_to_qmip = qmip_params['compound_to_qmip']
    qmip_to_compound = qmip_params['qmip_to_compound']
    available_compounds = qmip_params['available_compounds']

    # Configure stochastic evaluation mode
    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions={},  # stochastic rivals
    )

    # Seed RNGs
    np.random.seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)

    obs, _ = raw_env.reset()

    # Lap 0: choose starting compound
    action, t0_name = qmip_choose_start_compound(
        betas_dict, time_stop, raw_env.n_laps,
        qmip_to_compound, available_compounds, p=p,
    )
    obs, reward, terminated, truncated, info = raw_env.step(action)
    total_reward = reward

    # Track compounds used
    t_previous = [t0_name]
    current_compound = raw_env.agent_driver.compound
    pit_stops = 0

    done = terminated or truncated
    while not done:
        # Read state
        n_remaining = raw_env.laps_left
        t_current_name = compound_to_qmip.get(raw_env.agent_driver.compound, "Medium")
        w_current = raw_env.agent_driver.tyrelife
        gamma = 1 if (raw_env.vsc > 0 or raw_env.sc > 0) else 0
        n_total = raw_env.n_laps

        action = qmip_choose_action(
            betas_dict, time_stop, n_remaining,
            t_current_name, w_current, t_previous, n_total, gamma,
            qmip_to_compound, available_compounds, p=p,
        )

        obs, reward, terminated, truncated, info = raw_env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Track pit stops and compounds
        new_compound = raw_env.agent_driver.compound
        if new_compound != current_compound:
            pit_stops += 1
            current_compound = new_compound
            new_name = compound_to_qmip.get(new_compound, "Medium")
            if new_name not in t_previous:
                t_previous.append(new_name)

    changed_compound = info.get('has_changed_compound', False)
    raw_final_pos = info.get('final_position', raw_env.agent_position)
    final_pos = 20 if not changed_compound else raw_final_pos

    return SimResult(
        final_position=final_pos,
        total_reward=total_reward,
        pit_stops=pit_stops,
        changed_compound=changed_compound,
    )


# =============================================================================
# DRL RACE RUNNER
# =============================================================================

def run_single_race_drl(
    model,
    vec_env,
    raw_env,
    algo_name: str,
    seed: int,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_positions: List[int],
) -> SimResult:
    """Run a single race with a DRL agent (stochastic rivals, pre-generated SC/VSC)."""
    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions={},  # stochastic rivals
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)
    obs = vec_env.reset()

    use_lstm = (algo_name == 'RecurrentPPO')
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool) if use_lstm else None

    done = False
    total_reward = 0.0
    pit_stops = 0
    current_compound = None
    terminal_info = {}

    while not done:
        if use_lstm:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=False,
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=False)

        is_lap_zero = (raw_env.lap_number == 0)
        agent_compound = raw_env.agent_driver.compound

        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        done = done[0]

        if is_lap_zero:
            current_compound = raw_env.agent_driver.compound
        elif current_compound is None:
            current_compound = agent_compound
        elif agent_compound != current_compound:
            pit_stops += 1
            current_compound = agent_compound

        if done and 'final_position' in info[0]:
            terminal_info = info[0]

    changed_compound = terminal_info.get('has_changed_compound', False)
    raw_final_pos = terminal_info.get('final_position', raw_env.agent_position)
    final_pos = 20 if not changed_compound else raw_final_pos

    return SimResult(
        final_position=final_pos,
        total_reward=total_reward,
        pit_stops=pit_stops,
        changed_compound=changed_compound,
    )


# =============================================================================
# ALGORITHM EVALUATION
# =============================================================================

def evaluate_drl_algorithm(
    algo_name: str,
    model_path: str,
    loader: ModelLoader,
    gp_name: str,
    agent_driver: str,
    initial_positions: List[int],
    scenarios: List[Tuple[List[int], List[int]]],
    base_seed: int = 42,
) -> List[SimResult]:
    """Evaluate a single DRL algorithm over pre-generated scenarios."""
    raw_env = create_f1_env_all_drivers(
        gp=gp_name,
        driver=agent_driver,
        deterministic=False,
        yf_enabled=True,
        initial_positions=initial_positions,
        loader=loader,
        verbose=False,
    )

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
        vec_env = VecNormalize(
            vec_env, norm_reward=False,
            norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0,
        )
        vec_env.training = False

    AlgoClass = ALGO_CLASSES[algo_name]
    model = AlgoClass.load(model_path, env=vec_env)

    results = []
    n_sims = len(scenarios)
    for i, (laps_vsc, laps_sc) in enumerate(scenarios):
        result = run_single_race_drl(
            model, vec_env, raw_env, algo_name,
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


def evaluate_qmip(
    loader: ModelLoader,
    gp_name: str,
    agent_driver: str,
    initial_positions: List[int],
    scenarios: List[Tuple[List[int], List[int]]],
    qmip_params: dict,
    p: int = 3,
    base_seed: int = 42,
) -> List[SimResult]:
    """Evaluate QMIP over pre-generated scenarios."""
    raw_env = create_f1_env_all_drivers(
        gp=gp_name,
        driver=agent_driver,
        deterministic=False,
        yf_enabled=True,
        initial_positions=initial_positions,
        loader=loader,
        verbose=False,
    )

    results = []
    n_sims = len(scenarios)
    for i, (laps_vsc, laps_sc) in enumerate(scenarios):
        result = run_single_race_qmip(
            raw_env,
            seed=base_seed + i,
            laps_vsc=laps_vsc,
            laps_sc=laps_sc,
            initial_positions=initial_positions,
            qmip_params=qmip_params,
            p=p,
        )
        results.append(result)

        if (i + 1) % 100 == 0:
            positions = [r.final_position for r in results]
            print(f"      {i+1}/{n_sims} | Mean pos: {np.mean(positions):.2f}")

    return results


# =============================================================================
# PER-GP EVALUATION
# =============================================================================

def evaluate_gp(
    config_entry: dict,
    loader: ModelLoader,
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 42,
    p: int = 3,
) -> Dict[str, List[SimResult]]:
    """Evaluate all algorithms (DRL + QMIP) for a single GP."""
    gp_name = config_entry['gp_name']
    agent = config_entry['agent_driver']
    model_dir = config_entry['model_dir']

    gp_config = load_gp_config(gp_name)
    initial_positions = gp_config['initial_positions_list_2024']

    # Load GP models for SC/VSC probabilities
    gp_models = loader.load(gp_name)

    # Pre-generate SC/VSC scenarios (shared across all algorithms)
    scenarios = pregenenerate_race_scenarios(
        n_sims=n_simulations,
        n_laps=gp_models.n_laps,
        race_vsc_prob=gp_models.vsc_prob,
        race_sc_prob=gp_models.sc_prob,
        base_seed=base_seed * 1000,  # offset from per-sim seeds
    )

    positions_2024 = gp_config['initial_positions_2024']
    starting_pos = positions_2024.get(agent, '?')
    finishing_pos = config_entry['real_finishing_pos']

    print(f"\n{'=' * 70}")
    print(f"GP: {config_entry['display_name']}")
    print(f"{'=' * 70}")
    print(f"  Agent: {agent}  |  Start: P{starting_pos}  |  Real finish: P{finishing_pos}")
    print(f"  VSC prob: {gp_models.vsc_prob:.2f}  |  SC prob: {gp_models.sc_prob:.2f}")
    print(f"  Simulations: {n_simulations}  |  Stochastic SC/VSC + rivals")

    models_base = os.path.join(str(config.RL_AGENTS_DIR), model_dir)
    all_results: Dict[str, List[SimResult]] = {}

    # --- Evaluate DRL algorithms ---
    for algo in ALGO_ORDER:
        if algo == 'QMIP':
            continue

        model_path = os.path.join(models_base, algo, 'best_model.zip')
        if not os.path.exists(model_path):
            print(f"\n  [{algo}] Model not found at {model_path}, skipping.")
            continue

        print(f"\n{'~'*70}")
        print(f"  Evaluating {algo}...")
        print(f"{'~'*70}")

        results = evaluate_drl_algorithm(
            algo_name=algo,
            model_path=model_path,
            loader=loader,
            gp_name=gp_name,
            agent_driver=agent,
            initial_positions=initial_positions,
            scenarios=scenarios,
            base_seed=base_seed,
        )
        all_results[algo] = results

        positions = [r.final_position for r in results]
        dsq_count = sum(1 for r in results if not r.changed_compound)
        print(f"  {algo}: Mean P{np.mean(positions):.2f} +/- {np.std(positions):.2f}"
              f"  |  DSQ: {dsq_count}/{len(results)}")

    # --- Evaluate QMIP ---
    print(f"\n{'~'*70}")
    print(f"  Evaluating QMIP...")
    print(f"{'~'*70}")

    qmip_params = build_qmip_params(loader, gp_name)
    print(f"    betas: { {k: [f'{v[0]:.2f}', f'{v[1]:.4f}'] for k, v in qmip_params['betas_dict'].items()} }")
    print(f"    time_stop: {qmip_params['time_stop']:.2f}s")

    results = evaluate_qmip(
        loader=loader,
        gp_name=gp_name,
        agent_driver=agent,
        initial_positions=initial_positions,
        scenarios=scenarios,
        qmip_params=qmip_params,
        p=p,
        base_seed=base_seed,
    )
    all_results['QMIP'] = results

    positions = [r.final_position for r in results]
    dsq_count = sum(1 for r in results if not r.changed_compound)
    print(f"  QMIP: Mean P{np.mean(positions):.2f} +/- {np.std(positions):.2f}"
          f"  |  DSQ: {dsq_count}/{len(results)}")

    return all_results


# =============================================================================
# CACHE
# =============================================================================

def save_cache(
    all_data: List[dict],
    path: str = CACHE_FILE,
) -> None:
    rows = []
    for gp_idx, gp_data in enumerate(all_data):
        for algo, results in gp_data['results'].items():
            for r in results:
                rows.append({
                    'gp_idx': gp_idx,
                    'algo': algo,
                    'display_name': gp_data['display_name'],
                    'final_position': r.final_position,
                    'total_reward': r.total_reward,
                    'pit_stops': r.pit_stops,
                    'changed_compound': r.changed_compound,
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCache saved: {path}")


def load_cache(path: str = CACHE_FILE) -> List[dict]:
    df = pd.read_csv(path)
    all_data = []
    for gp_idx in sorted(df['gp_idx'].unique()):
        gp_df = df[df['gp_idx'] == gp_idx]
        display_name = gp_df['display_name'].iloc[0]
        results = {}
        for algo, algo_df in gp_df.groupby('algo'):
            results[algo] = [
                SimResult(
                    final_position=int(row.final_position),
                    total_reward=float(row.total_reward),
                    pit_stops=int(row.pit_stops),
                    changed_compound=bool(row.changed_compound),
                )
                for row in algo_df.itertuples()
            ]
        all_data.append({
            'display_name': display_name,
            'real_starting_pos': None,
            'real_finishing_pos': None,
            'results': results,
        })
    return all_data


# =============================================================================
# GRID BOXPLOT
# =============================================================================

def plot_grid_boxplot(
    all_gp_results: List[dict],
    save_path: Optional[str] = None,
):
    """Generate 3x3 grid boxplot comparing all algorithms (DRL + QMIP)."""
    n_gps = len(all_gp_results)
    nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, gp_data in enumerate(all_gp_results):
        ax = axes_flat[idx]
        results = gp_data['results']

        available = [a for a in ALGO_ORDER if a in results]
        position_data = [[r.final_position for r in results[a]] for a in available]
        tick_labels = [ALGO_ABBREV.get(a, a) for a in available]

        if position_data:
            bp = ax.boxplot(
                position_data,
                labels=tick_labels,
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

            for patch, algo in zip(bp['boxes'], available):
                color = ALGO_COLORS.get(algo, '#CCCCCC')
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

        # Reference lines
        starting_pos = gp_data.get('real_starting_pos')
        finishing_pos = gp_data.get('real_finishing_pos')

        if starting_pos is not None:
            offset = -0.15 if (finishing_pos is not None and starting_pos == finishing_pos) else 0
            ax.axhline(
                y=starting_pos + offset, color='red', linestyle='--',
                linewidth=1.5, zorder=0,
            )
        if finishing_pos is not None:
            ax.axhline(
                y=finishing_pos, color='blue', linestyle='--',
                linewidth=1.5, zorder=0,
            )

        ax.set_title(gp_data['display_name'], fontweight='bold')
        ax.set_ylim(0.5, 20.5)
        ax.set_yticks(range(2, 21, 2))
        ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.3, linestyle='-')
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('normal')
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('normal')

        if idx % ncols == 0:
            ax.set_ylabel('Final Position')

    for idx in range(n_gps, nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label='Actual Starting Position'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5,
               label='Actual Final Position'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markeredgecolor='red', markersize=8, label='Mean'),
    ]

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_elements),
        frameon=True,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nGrid plot saved to: {save_path}")

    plt.close(fig)


# =============================================================================
# LATEX TABLE
# =============================================================================

ALGO_DISPLAY = {
    'DQN': 'Double DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'Recurrent PPO',
    'QMIP': 'QMIP',
}


def generate_latex_table(all_gp_results: List[dict]) -> str:
    """Aggregate positions and points across all GPs and return a LaTeX table."""
    all_positions: Dict[str, List[int]] = {a: [] for a in ALGO_ORDER}

    for gp_data in all_gp_results:
        for algo, sim_results in gp_data['results'].items():
            for r in sim_results:
                all_positions[algo].append(r.final_position)

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{ccccc}')
    lines.append(r'\toprule')
    lines.append(r'\multicolumn{1}{l}{}               & \multicolumn{2}{c}{Position}   & \multicolumn{2}{c}{Points} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    lines.append(r'\multicolumn{1}{c}{Model}          &    $\mu$ & $\sigma$ &   $\mu$  & $\sigma$ \\ \midrule')

    for algo in ALGO_ORDER:
        positions = np.array(all_positions[algo])
        if len(positions) == 0:
            continue
        points = np.array([F1_POINTS.get(int(p), 0) for p in positions])

        mu_pos = np.mean(positions)
        std_pos = np.std(positions)
        mu_pts = np.mean(points)
        std_pts = np.std(points)

        display = ALGO_DISPLAY[algo]
        lines.append(
            rf'\multicolumn{{1}}{{c}}{{{display:<14}}} & '
            rf'{mu_pos:>8.2f}  & {std_pos:>8.2f}  & '
            rf'{mu_pts:>8.2f}  & {std_pts:>8.2f}  \\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Average positions and points obtained by each model (including QMIP), '
                 r'along with their standard deviations on selected circuits after '
                 r'500 stochastic simulations per model for the 2024 Season.}')
    lines.append(r'\label{tab:tab_qmip_vs_drl_stochastic_2024}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def is_entry_complete(entry: dict) -> bool:
    if entry['real_finishing_pos'] is None:
        return False
    for driver, stops in entry['rival_strategies'].items():
        for lap, compound in stops.items():
            if compound is None:
                return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate QMIP vs DRL under stochastic SC/VSC conditions',
    )
    parser.add_argument(
        '--n-sims', type=int, default=N_SIMULATIONS,
        help=f'Number of simulations per algorithm (default: {N_SIMULATIONS})',
    )
    parser.add_argument(
        '--gps', nargs='+', default=None,
        help='Filter by model_dir name (e.g., Bahrain_ALO Belgian_NOR)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed (default: 42)',
    )
    parser.add_argument(
        '--max-pits', type=int, default=3,
        help='Max pit stops for QMIP (default: 3)',
    )
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Only plot from cached results',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-evaluation even if cache exists',
    )
    args = parser.parse_args()

    save_dir = str(config.RL_AGENTS_DIR)

    # --- Plot-only mode ---
    if args.plot_only:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: Cache file not found: {CACHE_FILE}")
            sys.exit(1)
        print("Loading cached results...")
        all_gp_results = load_cache(CACHE_FILE)

        # Restore starting/finishing positions from GP_CONFIGS
        for i, gp_data in enumerate(all_gp_results):
            if i < len(GP_CONFIGS):
                entry = GP_CONFIGS[i]
                gp_config = load_gp_config(entry['gp_name'])
                gp_data['real_starting_pos'] = gp_config['initial_positions_2024'].get(
                    entry['agent_driver'],
                )
                gp_data['real_finishing_pos'] = entry['real_finishing_pos']

        plot_grid_boxplot(
            all_gp_results,
            save_path=os.path.join(save_dir, 'qmip_vs_drl_stochastic_boxplot.pdf'),
        )
        plot_grid_boxplot(
            all_gp_results,
            save_path=os.path.join(save_dir, 'qmip_vs_drl_stochastic_boxplot.png'),
        )

        latex = generate_latex_table(all_gp_results)
        tex_path = os.path.join(save_dir, 'qmip_vs_drl_stochastic_table.tex')
        with open(tex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {tex_path}")

        sys.exit(0)

    # --- Full evaluation ---
    configs = GP_CONFIGS
    if args.gps:
        configs = [c for c in configs if c['model_dir'] in args.gps]
        if not configs:
            print(f"No matching GP configs for: {args.gps}")
            print(f"Available: {[c['model_dir'] for c in GP_CONFIGS]}")
            sys.exit(1)

    print("=" * 70)
    print("EVALUATION: QMIP vs DRL (Stochastic SC/VSC + Rivals)")
    print("=" * 70)
    print(f"  GPs to evaluate: {len(configs)}")
    print(f"  Simulations per algorithm: {args.n_sims}")
    print(f"  Base seed: {args.seed}")
    print(f"  QMIP max pits: {args.max_pits}")
    print("=" * 70)

    loader = ModelLoader()
    all_gp_results = []

    for entry in configs:
        if not is_entry_complete(entry):
            print(f"\n  WARNING: Skipping {entry['model_dir']} -- incomplete data.")
            continue

        results = evaluate_gp(
            entry, loader,
            n_simulations=args.n_sims,
            base_seed=args.seed,
            p=args.max_pits,
        )

        if not results:
            print(f"\n  WARNING: No results for {entry['model_dir']}, skipping.")
            continue

        gp_config = load_gp_config(entry['gp_name'])
        starting_pos = gp_config['initial_positions_2024'].get(entry['agent_driver'])

        all_gp_results.append({
            'display_name': entry['display_name'],
            'real_starting_pos': starting_pos,
            'real_finishing_pos': entry['real_finishing_pos'],
            'results': results,
        })

    if not all_gp_results:
        print("No results to plot.")
        sys.exit(1)

    # Save cache
    save_cache(all_gp_results)

    # Plot
    plot_grid_boxplot(
        all_gp_results,
        save_path=os.path.join(save_dir, 'qmip_vs_drl_stochastic_boxplot.pdf'),
    )
    plot_grid_boxplot(
        all_gp_results,
        save_path=os.path.join(save_dir, 'qmip_vs_drl_stochastic_boxplot.png'),
    )

    # LaTeX table
    latex = generate_latex_table(all_gp_results)
    tex_path = os.path.join(save_dir, 'qmip_vs_drl_stochastic_table.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to: {tex_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for gp_data in all_gp_results:
        print(f"\n  {gp_data['display_name']}:")
        for algo in ALGO_ORDER:
            if algo in gp_data['results']:
                positions = [r.final_position for r in gp_data['results'][algo]]
                print(f"    {ALGO_ABBREV[algo]:>5}: "
                      f"Mean P{np.mean(positions):.1f} "
                      f"+/- {np.std(positions):.1f}  "
                      f"Median P{int(np.median(positions))}")
