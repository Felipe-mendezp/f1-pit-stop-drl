import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

# Allow importing sibling scripts by name
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

"""
Evaluate trained RL models + QMIP against real 2024 race data for ALL Grand Prix.

Reproduces Figure 7 of the paper (real-race-realizations boxplot, including
QMIP overlay). Same setup as evaluate_all_gps.py (real SC/VSC events and real
rival pit strategies per GP), but adds QMIP as a 6th model so the boxplot
shows 6 boxes per subplot: DQN, A2C, TRPO, PPO, RecurrentPPO, QMIP.

Usage:
    python scripts/evaluate_all_gps_with_qmip.py
    python scripts/evaluate_all_gps_with_qmip.py --n-sims 50
    python scripts/evaluate_all_gps_with_qmip.py --gps Bahrain_ALO Belgian_NOR
    python scripts/evaluate_all_gps_with_qmip.py --plot-only
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
from typing import Dict, List, Optional

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from evaluate_all_gps import (
    GP_CONFIGS,
    NORM_OBS_KEYS,
    SimResult,
    F1_POINTS,
    load_gp_config,
    build_rival_strategies,
    evaluate_algorithm,
    is_entry_complete,
)
from evaluate_qmip_vs_drl import (
    build_qmip_params,
    qmip_choose_start_compound,
    qmip_choose_action,
)
from model_loader import ModelLoader

# Match the typography used in the paper figures. Must come AFTER imports
# because evaluate_all_gps sets font.family at module scope.
mpl.rcParams.update({
    'font.family':           'sans-serif',
    'axes.titlesize':        17,
    'axes.labelsize':        18,
    'xtick.labelsize':       15,
    'ytick.labelsize':       15,
    'legend.fontsize':       16,
    'legend.title_fontsize': 16,
})


N_SIMULATIONS = 500

CACHE_FILE = os.path.join(str(config.RL_AGENTS_DIR), 'all_gps_2024_qmip_results.csv')

ALGO_ORDER = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']

ALGO_COLORS = {
    'DQN':          '#0072B2',
    'A2C':          '#E69F00',
    'TRPO':         '#009E73',
    'PPO':          '#CC79A7',
    'RecurrentPPO': '#56B4E9',
    'QMIP':         '#D55E00',
}

ALGO_ABBREV = {
    'DQN': 'DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'RPPO',
    'QMIP': 'QMIP',
}

ALGO_DISPLAY = {
    'DQN': 'Double DQN',
    'A2C': 'A2C',
    'TRPO': 'TRPO',
    'PPO': 'PPO',
    'RecurrentPPO': 'Recurrent PPO',
    'QMIP': 'QMIP',
}

DRL_ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']


# =============================================================================
# QMIP RACE RUNNER -- REAL RIVAL ACTIONS + REAL SC/VSC
# =============================================================================

def run_single_race_qmip_real(
    raw_env,
    seed: int,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_positions: List[int],
    rival_actions: dict,
    initial_compounds: Dict[str, int],
    qmip_params: dict,
    p: int = 3,
) -> SimResult:
    """Run one QMIP race using real 2024 SC/VSC events and real rival pit strategies."""
    betas_dict = qmip_params['betas_dict']
    time_stop = qmip_params['time_stop']
    compound_to_qmip = qmip_params['compound_to_qmip']
    qmip_to_compound = qmip_params['qmip_to_compound']
    available_compounds = qmip_params['available_compounds']

    raw_env.set_evaluation_mode(
        eval_mode=True,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_positions=initial_positions,
        rival_actions=rival_actions,
    )

    np.random.seed(seed)
    raw_env.np_random = np.random.default_rng(seed)
    raw_env.action_space.seed(seed)

    obs, _ = raw_env.reset()

    for rival in raw_env.rival_drivers:
        compound = initial_compounds.get(rival.name)
        if compound is not None:
            rival.compound = compound
            rival.initial_compound = compound

    action, t0_name = qmip_choose_start_compound(
        betas_dict, time_stop, raw_env.n_laps,
        qmip_to_compound, available_compounds, p=p,
    )
    obs, reward, terminated, truncated, info = raw_env.step(action)
    total_reward = reward

    t_previous = [t0_name]
    current_compound = raw_env.agent_driver.compound
    pit_stops = 0

    done = terminated or truncated
    while not done:
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


def evaluate_qmip_real(
    loader: ModelLoader,
    gp_name: str,
    agent_driver: str,
    initial_positions: List[int],
    rival_strategies: dict,
    laps_vsc: List[int],
    laps_sc: List[int],
    initial_compounds: Dict[str, int],
    qmip_params: dict,
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 0,
    p: int = 3,
) -> List[SimResult]:
    """Evaluate QMIP over N simulations for a given GP with real rival actions."""
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
    for i in range(n_simulations):
        result = run_single_race_qmip_real(
            raw_env,
            seed=base_seed + i,
            laps_vsc=laps_vsc,
            laps_sc=laps_sc,
            initial_positions=initial_positions,
            rival_actions=rival_strategies,
            initial_compounds=initial_compounds,
            qmip_params=qmip_params,
            p=p,
        )
        results.append(result)

        if (i + 1) % 25 == 0:
            positions = [r.final_position for r in results]
            print(f"    {i+1}/{n_simulations}  |  "
                  f"Mean pos: {np.mean(positions):.2f}  |  "
                  f"Best: P{min(positions)}  |  "
                  f"DSQ: {sum(1 for r in results if not r.changed_compound)}")

    return results


# =============================================================================
# PER-GP EVALUATION -- DRL (real) + QMIP (real)
# =============================================================================

def evaluate_gp_with_qmip(
    config_entry: dict,
    loader: ModelLoader,
    n_simulations: int = N_SIMULATIONS,
    base_seed: int = 0,
    p: int = 3,
) -> Dict[str, List[SimResult]]:
    """Evaluate 5 DRL algorithms + QMIP for a single GP/driver combination."""
    gp_config = load_gp_config(config_entry['gp_name'])
    rival_strategies, initial_compounds = build_rival_strategies(config_entry)

    initial_positions = gp_config['initial_positions_list_2024']
    laps_vsc = gp_config['laps_vsc_2024']
    laps_sc = gp_config['laps_sc_2024']

    positions_2024 = gp_config['initial_positions_2024']
    agent = config_entry['agent_driver']
    starting_pos = positions_2024.get(agent, '?')
    finishing_pos = config_entry['real_finishing_pos']

    model_dir = config_entry['model_dir']
    models_base = os.path.join(str(config.RL_AGENTS_DIR), model_dir)

    print(f"\n{'=' * 70}")
    print(f"GP: {config_entry['display_name']}")
    print(f"{'=' * 70}")
    print(f"  Agent: {agent}  |  Start: P{starting_pos}  |  Real finish: P{finishing_pos}")
    print(f"  SC laps: {laps_sc}  |  VSC laps: {laps_vsc}")
    print(f"  Simulations per algorithm: {n_simulations}")

    all_results: Dict[str, List[SimResult]] = {}

    # --- DRL algorithms ---
    for algo in DRL_ALGOS:
        model_path = os.path.join(models_base, algo, 'best_model.zip')

        if not os.path.exists(model_path):
            print(f"\n  [{algo}] Model not found at {model_path}, skipping.")
            continue

        print(f"\n{'-' * 70}")
        print(f"  Evaluating {algo}...")
        print(f"{'-' * 70}")

        results = evaluate_algorithm(
            algo_name=algo,
            model_path=model_path,
            loader=loader,
            gp_name=config_entry['gp_name'],
            agent_driver=agent,
            initial_positions=initial_positions,
            rival_strategies=rival_strategies,
            laps_vsc=laps_vsc,
            laps_sc=laps_sc,
            initial_compounds=initial_compounds,
            n_simulations=n_simulations,
            base_seed=base_seed,
        )
        all_results[algo] = results

        positions = [r.final_position for r in results]
        dsq_count = sum(1 for r in results if not r.changed_compound)
        print(f"  {algo}: Mean P{np.mean(positions):.2f} +/- {np.std(positions):.2f}"
              f"  |  DSQ: {dsq_count}/{len(results)}")

    # --- QMIP ---
    print(f"\n{'-' * 70}")
    print(f"  Evaluating QMIP...")
    print(f"{'-' * 70}")

    qmip_params = build_qmip_params(loader, config_entry['gp_name'])
    print(f"    betas: { {k: [f'{v[0]:.2f}', f'{v[1]:.4f}'] for k, v in qmip_params['betas_dict'].items()} }")
    print(f"    time_stop: {qmip_params['time_stop']:.2f}s")

    qmip_results = evaluate_qmip_real(
        loader=loader,
        gp_name=config_entry['gp_name'],
        agent_driver=agent,
        initial_positions=initial_positions,
        rival_strategies=rival_strategies,
        laps_vsc=laps_vsc,
        laps_sc=laps_sc,
        initial_compounds=initial_compounds,
        qmip_params=qmip_params,
        n_simulations=n_simulations,
        base_seed=base_seed,
        p=p,
    )
    all_results['QMIP'] = qmip_results

    positions = [r.final_position for r in qmip_results]
    dsq_count = sum(1 for r in qmip_results if not r.changed_compound)
    print(f"  QMIP: Mean P{np.mean(positions):.2f} +/- {np.std(positions):.2f}"
          f"  |  DSQ: {dsq_count}/{len(qmip_results)}")

    return all_results


# =============================================================================
# GRID BOXPLOT -- 6 algorithms
# =============================================================================

def plot_grid_boxplot(
    all_gp_results: List[dict],
    save_path: Optional[str] = None,
):
    """Generate 3x3 grid boxplot showing DRL + QMIP per GP with real race data."""
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

            for patch, algo in zip(bp['boxes'], available):
                color = ALGO_COLORS.get(algo, '#CCCCCC')
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

        starting_pos = gp_data.get('real_starting_pos')
        finishing_pos = gp_data.get('real_finishing_pos')

        if starting_pos is not None:
            offset = -0.15 if (finishing_pos is not None and starting_pos == finishing_pos) else 0
            ax.axhline(y=starting_pos + offset, color='red', linestyle='--',
                       linewidth=1.5, zorder=0)
        if finishing_pos is not None:
            ax.axhline(y=finishing_pos, color='blue', linestyle='--',
                       linewidth=1.5, zorder=0)

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

def generate_latex_table(all_gp_results: List[dict]) -> str:
    """Aggregate positions and points across all GPs (6 algorithms)."""
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
        points = np.array([F1_POINTS.get(int(pos), 0) for pos in positions])

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
    lines.append(r'\caption{Average positions and points per model (including QMIP) '
                 r'with real 2024 SC/VSC and real rival pit strategies, '
                 r'across 500 simulations per model on selected 2024 GPs.}')
    lines.append(r'\label{tab:tab_all_gps_2024_with_qmip}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# =============================================================================
# CACHE
# =============================================================================

def save_cache(all_data: List[dict], path: str = CACHE_FILE) -> None:
    """Persist simulation results to CSV so plots can be regenerated without re-running."""
    rows = []
    for gp_idx, gp_data in enumerate(all_data):
        for algo, results in gp_data['results'].items():
            for r in results:
                rows.append({
                    'gp_idx': gp_idx,
                    'algo': algo,
                    'display_name': gp_data['display_name'],
                    'real_starting_pos': gp_data.get('real_starting_pos'),
                    'real_finishing_pos': gp_data.get('real_finishing_pos'),
                    'final_position': r.final_position,
                    'total_reward': r.total_reward,
                    'pit_stops': r.pit_stops,
                    'changed_compound': r.changed_compound,
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCache saved: {path}")


def load_cache(path: str = CACHE_FILE) -> List[dict]:
    """Rebuild all_gp_results from a CSV cache."""
    df = pd.read_csv(path)
    all_data = []
    for gp_idx in sorted(df['gp_idx'].unique()):
        gp_df = df[df['gp_idx'] == gp_idx]
        display_name = gp_df['display_name'].iloc[0]
        starting_raw = gp_df['real_starting_pos'].iloc[0]
        finishing_raw = gp_df['real_finishing_pos'].iloc[0]
        starting_pos = None if pd.isna(starting_raw) else int(starting_raw)
        finishing_pos = None if pd.isna(finishing_raw) else int(finishing_raw)
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
            'real_starting_pos': starting_pos,
            'real_finishing_pos': finishing_pos,
            'results': results,
        })
    return all_data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate RL agents + QMIP against real 2024 race data for all GPs',
    )
    parser.add_argument('--n-sims', type=int, default=N_SIMULATIONS,
                        help=f'Simulations per algorithm (default: {N_SIMULATIONS})')
    parser.add_argument('--gps', nargs='+', default=None,
                        help='Filter by model_dir (e.g., Bahrain_ALO Belgian_NOR)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed (default: 0)')
    parser.add_argument('--max-pits', type=int, default=3,
                        help='Max pit stops for QMIP (default: 3)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Load cached results and regenerate plots/table without re-running simulations')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all simulations even if cache exists')
    args = parser.parse_args()

    save_dir = str(config.RL_AGENTS_DIR)

    def regenerate_outputs(all_gp_results):
        output_pdf = os.path.join(save_dir, 'all_gps_2024_qmip_boxplot.pdf')
        output_png = os.path.join(save_dir, 'all_gps_2024_qmip_boxplot.png')
        plot_grid_boxplot(all_gp_results, save_path=output_pdf)
        plot_grid_boxplot(all_gp_results, save_path=output_png)

        latex_table = generate_latex_table(all_gp_results)
        table_path = os.path.join(save_dir, 'all_gps_2024_qmip_table.tex')
        with open(table_path, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {table_path}")

    # --- Plot-only mode ---
    if args.plot_only:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: Cache file not found: {CACHE_FILE}")
            print("Run without --plot-only to generate simulations first.")
            sys.exit(1)
        print(f"Loading cached results from: {CACHE_FILE}")
        all_gp_results = load_cache(CACHE_FILE)
        regenerate_outputs(all_gp_results)
        sys.exit(0)

    # --- Reuse cache unless --force ---
    if os.path.exists(CACHE_FILE) and not args.force:
        print(f"Cache exists at {CACHE_FILE}. Use --force to re-run or --plot-only to skip evaluation.")
        print("Loading cached results...")
        all_gp_results = load_cache(CACHE_FILE)
        regenerate_outputs(all_gp_results)
        sys.exit(0)

    configs = GP_CONFIGS
    if args.gps:
        configs = [c for c in configs if c['model_dir'] in args.gps]
        if not configs:
            print(f"No matching GP configs for: {args.gps}")
            print(f"Available: {[c['model_dir'] for c in GP_CONFIGS]}")
            sys.exit(1)

    print("=" * 70)
    print("EVALUATION: RL Agents + QMIP vs Real 2024 Race Results")
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

        results = evaluate_gp_with_qmip(
            entry, loader,
            n_simulations=args.n_sims,
            base_seed=args.seed,
            p=args.max_pits,
        )

        if not results:
            print(f"\n  WARNING: No models found for {entry['model_dir']}, skipping.")
            continue

        gp_config = load_gp_config(entry['gp_name'])
        starting_pos = gp_config['initial_positions_2024'].get(entry['agent_driver'])

        all_gp_results.append({
            'display_name': entry['display_name'],
            'real_starting_pos': starting_pos,
            'real_finishing_pos': entry['real_finishing_pos'],
            'results': results,
        })

        # Per-GP comparison table
        print(f"\n{'=' * 70}")
        print(f"COMPARISON TABLE -- {entry['display_name']}")
        print(f"{'=' * 70}")
        finishing = entry['real_finishing_pos']
        print(f"{'Algorithm':<15} {'Mean Pos':<12} {'Median':<8} {'Best':<6} "
              f"{'Pits':<6} {'Beat Real':<10}")
        print("-" * 70)
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            res = results[algo]
            positions = [r.final_position for r in res]
            mean_pits = np.mean([r.pit_stops for r in res])
            beat = sum(1 for pos in positions if pos < finishing)
            print(f"{algo:<15} "
                  f"{np.mean(positions):>5.2f} +/- {np.std(positions):<4.1f} "
                  f"P{int(np.median(positions)):<5} "
                  f"P{min(positions):<4} "
                  f"{mean_pits:<5.1f} "
                  f"{beat:>3}/{len(res)} ({beat/len(res)*100:>4.0f}%)")
        print("-" * 70)

    if not all_gp_results:
        print("\nNo complete GP evaluations.")
        sys.exit(1)

    save_cache(all_gp_results)
    regenerate_outputs(all_gp_results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"  GPs evaluated: {len(all_gp_results)}")
    print(f"  GPs skipped: {len(configs) - len(all_gp_results)}")
    print("=" * 70)
