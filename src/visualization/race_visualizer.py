"""
Race Visualization for F1 Multi-Driver Environment

Creates visualizations showing:
- Tire stints for all 20 drivers
- Pit stops
- VSC/SC events
- Race progression
"""

import config

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from environment.f1_env_all_drivers import create_f1_env_all_drivers


@dataclass
class RaceEvent:
    """Records a single event during the race."""
    lap: int
    driver: str
    event_type: str  # 'stint', 'pitstop', 'vsc', 'sc'
    compound: Optional[int] = None
    position: Optional[int] = None


@dataclass
class DriverStint:
    """Records a tire stint for a driver."""
    driver: str
    compound: int
    start_lap: int
    end_lap: int
    start_position: int
    end_position: int


@dataclass
class RaceHistory:
    """Complete race history for visualization."""
    stints: List[DriverStint] = field(default_factory=list)
    vsc_laps: List[int] = field(default_factory=list)
    sc_laps: List[int] = field(default_factory=list)
    driver_positions: Dict[str, List[int]] = field(default_factory=dict)
    driver_names: List[str] = field(default_factory=list)
    total_laps: int = 0
    gp_name: str = ""


def simulate_race_with_history(
    gp: str,
    driver: str,
    agent_strategy: List[Tuple[int, int]],  # [(compound, lap_to_pit), ...]
    deterministic: bool = True,
    seed: int = 42,
    yf_enabled: bool = True,
    verbose: bool = True,
    initial_positions: Optional[List[int]] = None
) -> RaceHistory:
    """
    Simulate a complete race and record full history.

    Args:
        gp: Grand Prix name
        driver: Agent driver name
        agent_strategy: List of (compound, lap) tuples for agent pit stops
            e.g., [(2, 0), (1, 25)] means start with C2, pit on lap 25 to C1
        deterministic: Fixed random seed for reproducibility
        seed: Random seed
        yf_enabled: Enable VSC/SC events
        verbose: Print race progress
        initial_positions: Optional list of 20 starting positions (1-20) for each driver.
            If None, positions are randomized.

    Returns:
        RaceHistory with complete race data
    """
    env = create_f1_env_all_drivers(
        gp=gp,
        driver=driver,
        deterministic=deterministic,
        yf_enabled=yf_enabled,
        initial_positions=initial_positions,
        verbose=False
    )

    history = RaceHistory()
    history.gp_name = gp
    history.total_laps = env.n_laps
    history.driver_names = [d.name for d in sorted(env.drivers, key=lambda x: x.initial_pos)]

    # Track current stints for each driver
    current_stints: Dict[str, Dict] = {}

    # Reset environment
    obs, _ = env.reset(seed=seed)

    # Initial positions
    for d in env.drivers:
        history.driver_positions[d.name] = [d.position]

    # Lap 0: Choose starting compound
    starting_compound = agent_strategy[0][0] if agent_strategy else 2
    action = env.available_compounds.index(starting_compound) + 1
    obs, reward, done, _, _ = env.step(action)

    if verbose:
        print(f"Starting Race: {gp}")
        print(f"Agent: {driver}, Starting compound: C{starting_compound}")
        print(f"Starting position: {env.agent_position}")
        print("=" * 70)

    # Initialize stints for all drivers
    for d in env.drivers:
        current_stints[d.name] = {
            'compound': d.compound,
            'start_lap': 1,
            'start_position': d.position
        }

    # Prepare pit strategy
    pit_schedule = {}
    for compound, lap in agent_strategy[1:]:
        pit_schedule[lap] = compound

    # Run race
    lap = 1
    while not done and lap <= env.n_laps:
        # Check for agent pit stop
        if lap in pit_schedule:
            action = env.available_compounds.index(pit_schedule[lap]) + 1
            if verbose:
                print(f"Lap {lap}: Agent PIT STOP to C{pit_schedule[lap]}")
        else:
            action = 0  # No pit

        # Execute step
        obs, reward, done, _, _ = env.step(action)

        # Record VSC/SC
        if env.vsc > 0:
            history.vsc_laps.append(lap)
        if env.sc > 0:
            history.sc_laps.append(lap)

        # Update positions
        for d in env.drivers:
            history.driver_positions[d.name].append(d.position)

            # Check for stint change (pit stop)
            if d.compound != current_stints[d.name]['compound']:
                # End previous stint
                old_stint = current_stints[d.name]
                history.stints.append(DriverStint(
                    driver=d.name,
                    compound=old_stint['compound'],
                    start_lap=old_stint['start_lap'],
                    end_lap=lap - 1,
                    start_position=old_stint['start_position'],
                    end_position=d.position
                ))

                # Start new stint
                current_stints[d.name] = {
                    'compound': d.compound,
                    'start_lap': lap,
                    'start_position': d.position
                }

                if verbose and d.name == driver:
                    print(f"Lap {lap}: {d.name} changed to C{d.compound}, pos={d.position}")

        if verbose and lap % 10 == 0:
            print(f"Lap {lap}: Agent pos={env.agent_position}")

        lap += 1

    # Finalize remaining stints
    for d in env.drivers:
        if d.name in current_stints:
            stint = current_stints[d.name]
            history.stints.append(DriverStint(
                driver=d.name,
                compound=stint['compound'],
                start_lap=stint['start_lap'],
                end_lap=env.n_laps,
                start_position=stint['start_position'],
                end_position=d.position
            ))

    if verbose:
        print("=" * 70)
        print(f"Race Complete!")
        print(f"Final position: {env.agent_position}")
        print(f"Total time: {env.agent_total_time:.2f}s")
        print(f"VSC laps: {len(history.vsc_laps)}, SC laps: {len(history.sc_laps)}")

    return history


def plot_race_stints(
    history: RaceHistory,
    figsize: Tuple[int, int] = (16, 10),
    show_positions: bool = False,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive race stint visualization.

    Args:
        history: RaceHistory object from simulate_race_with_history
        figsize: Figure size (width, height)
        show_positions: Show position changes in separate subplot
        save_path: Path to save figure (if None, just displays)
    """
    # Compound colors (consistent with F1 color scheme)
    compound_colors = {
        1: '#FF0000',  # C1 - Red (hardest)
        2: '#FFFFFF',  # C2 - White
        3: '#FFFF00',  # C3 - Yellow
        4: '#00FF00',  # C4 - Green
        5: '#0000FF',  # C5 - Blue (softest)
    }

    # Create figure
    n_subplots = 2 if show_positions else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': [3, 1]} if show_positions else None)

    if not show_positions:
        axes = [axes]

    ax_stints = axes[0]

    # Sort drivers by final position
    final_positions = {name: positions[-1] for name, positions in history.driver_positions.items()}
    sorted_drivers = sorted(final_positions.keys(), key=lambda x: final_positions[x])

    # Create y-axis mapping (position on plot)
    driver_y_positions = {name: i for i, name in enumerate(sorted_drivers)}

    # Plot stints
    for stint in history.stints:
        y_pos = driver_y_positions[stint.driver]
        width = stint.end_lap - stint.start_lap
        color = compound_colors.get(stint.compound, '#808080')

        # Draw stint rectangle
        rect = Rectangle(
            (stint.start_lap, y_pos - 0.4),
            width,
            0.8,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax_stints.add_patch(rect)

        # Add compound label in the middle of stint
        if width > 3:
            ax_stints.text(
                stint.start_lap + width / 2,
                y_pos,
                f'C{stint.compound}',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold'
            )

    # Plot VSC periods
    for lap in history.vsc_laps:
        ax_stints.axvspan(lap - 0.5, lap + 0.5, alpha=0.2, color='yellow', zorder=0)

    # Plot SC periods
    for lap in history.sc_laps:
        ax_stints.axvspan(lap - 0.5, lap + 0.5, alpha=0.3, color='orange', zorder=0)

    # Formatting
    ax_stints.set_ylim(-0.5, len(sorted_drivers) - 0.5)
    ax_stints.set_xlim(0, history.total_laps + 1)
    ax_stints.set_yticks(range(len(sorted_drivers)))
    ax_stints.set_yticklabels([
        f"P{final_positions[d]} - {d.replace('Driver_', '')}"
        for d in sorted_drivers
    ])
    ax_stints.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax_stints.set_ylabel('Driver (Final Position)', fontsize=12, fontweight='bold')
    ax_stints.set_title(f'{history.gp_name} - Tire Strategy Overview',
                        fontsize=14, fontweight='bold', pad=20)
    ax_stints.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax_stints.invert_yaxis()

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=compound_colors[c], edgecolor='black', label=f'C{c}')
        for c in sorted(compound_colors.keys())
    ]
    if history.vsc_laps:
        legend_elements.append(mpatches.Patch(facecolor='yellow', alpha=0.2, label='VSC'))
    if history.sc_laps:
        legend_elements.append(mpatches.Patch(facecolor='orange', alpha=0.3, label='SC'))

    ax_stints.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        title='Compounds & Events'
    )

    # Optional: Position progression plot
    if show_positions:
        ax_pos = axes[1]

        # Plot position changes for top 10 finishers
        for i, driver in enumerate(sorted_drivers[:10]):
            positions = history.driver_positions[driver]
            ax_pos.plot(
                range(len(positions)),
                positions,
                marker='o' if i < 3 else None,
                markersize=3 if i < 3 else 0,
                linewidth=2 if i < 3 else 1,
                alpha=0.8 if i < 3 else 0.5,
                label=driver.replace('Driver_', '')
            )

        ax_pos.set_xlabel('Lap Number', fontsize=10, fontweight='bold')
        ax_pos.set_ylabel('Position', fontsize=10, fontweight='bold')
        ax_pos.set_title('Position Changes (Top 10 Finishers)', fontsize=12, fontweight='bold')
        ax_pos.invert_yaxis()
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc='upper left', fontsize=8, ncol=2)
        ax_pos.set_ylim(20.5, 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig


def print_race_summary(history: RaceHistory):
    """Print a text summary of the race."""
    print("\n" + "=" * 70)
    print(f"RACE SUMMARY: {history.gp_name}")
    print("=" * 70)

    # Final positions
    final_positions = {name: positions[-1] for name, positions in history.driver_positions.items()}
    sorted_drivers = sorted(final_positions.keys(), key=lambda x: final_positions[x])

    print("\nFINAL CLASSIFICATION:")
    print("-" * 70)
    for i, driver in enumerate(sorted_drivers[:10], 1):
        stints = [s for s in history.stints if s.driver == driver]
        compounds_used = sorted(set(s.compound for s in stints))
        pit_stops = max(0, len(stints) - 1)  # -1 because first stint doesn't count
        print(f"P{i:2d}  {driver:20s}  Pit Stops: {pit_stops}  "
              f"Compounds: {', '.join(f'C{c}' for c in compounds_used)}")

    print("\nSAFETY CAR EVENTS:")
    print("-" * 70)
    if history.vsc_laps:
        print(f"VSC: {len(set(history.vsc_laps))} lap(s) - Laps {sorted(set(history.vsc_laps))}")
    else:
        print("VSC: None")

    if history.sc_laps:
        print(f"SC:  {len(set(history.sc_laps))} lap(s) - Laps {sorted(set(history.sc_laps))}")
    else:
        print("SC:  None")

    print("\nAGENT STRATEGY DETAILS:")
    print("-" * 70)
    # Try to find agent driver (VER by default, or first driver in sorted list)
    agent_driver = None
    for d in history.driver_names:
        if 'VER' in d:
            agent_driver = d
            break
    if agent_driver is None:
        agent_driver = sorted_drivers[0]

    agent_stints = [s for s in history.stints if s.driver == agent_driver]

    if agent_stints:
        for i, stint in enumerate(agent_stints, 1):
            print(f"Stint {i}: C{stint.compound}, Laps {stint.start_lap}-{stint.end_lap} "
                  f"({stint.end_lap - stint.start_lap + 1} laps), "
                  f"Pos {stint.start_position}->{stint.end_position}")
    else:
        print(f"No stint data available for {agent_driver}")

    print("=" * 70 + "\n")


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Define agent strategy
    # Format: [(starting_compound, 0), (compound_after_pit1, lap_to_pit), ...]
    agent_strategy = [
        (2, 0),   # Start with C2
        (1, 25),  # Pit on lap 25, change to C1
    ]

    # Simulate race
    print("Simulating Bahrain Grand Prix...")
    history = simulate_race_with_history(
        gp='Bahrain Grand Prix',
        driver='Driver_VER',
        agent_strategy=agent_strategy,
        deterministic=True,
        seed=42,
        yf_enabled=True,
        verbose=True
    )

    # Print summary
    print_race_summary(history)

    # Create visualization
    print("\nGenerating visualization...")
    plot_race_stints(
        history,
        figsize=(18, 12),
        show_positions=True,
        save_path=str(config.ROOT_DIR / 'race_stints_example.png')
    )
