"""
Generate bahrain_rival_stints_grid.png
2x2 grid of 4 simulations showing tire stints for all 20 drivers.
Agent: VER with a C2->C1 1-stop strategy. Rivals use logit models.
"""

import config

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from visualization.race_visualizer import simulate_race_with_history, RaceHistory

GP = 'Bahrain Grand Prix'
AGENT = 'Driver_VER'
AGENT_STRATEGY = [
    (2, 0),   # Start with C2 (Medium)
    (1, 25),  # Pit on lap 25, change to C1 (Hard)
]

SEEDS = [42, 1233, 7, 2024]

COMPOUND_COLORS = {
    1: '#E8443A',  # C1 Hard - Red
    2: '#F5F5F0',  # C2 Medium - White/light
    3: '#F5A623',  # C3 Soft - Orange/Yellow
    4: '#7ED321',  # C4 - Green
    5: '#4A90D9',  # C5 - Blue
}
COMPOUND_EDGE = {
    1: '#C0392B',
    2: '#AAAAAA',
    3: '#E09000',
    4: '#5CA015',
    5: '#2E6DA4',
}
COMPOUND_LABELS = {1: 'C1 (Hard)', 2: 'C2 (Medium)', 3: 'C3 (Soft)', 4: 'C4', 5: 'C5'}


def plot_single_sim(ax, history: RaceHistory, sim_idx: int, seed: int):
    """Plot a single simulation's stint chart on the given axes."""
    # Sort drivers by final position
    final_positions = {name: positions[-1] for name, positions in history.driver_positions.items()}
    sorted_drivers = sorted(final_positions.keys(), key=lambda x: final_positions[x])
    driver_y = {name: i for i, name in enumerate(sorted_drivers)}

    total_pits = sum(1 for s in history.stints if s.start_lap > 1)
    sc_laps = len(set(history.sc_laps))
    vsc_laps = len(set(history.vsc_laps))

    for stint in history.stints:
        y = driver_y[stint.driver]
        width = stint.end_lap - stint.start_lap
        color = COMPOUND_COLORS.get(stint.compound, '#808080')
        edge = COMPOUND_EDGE.get(stint.compound, '#555555')

        is_agent = stint.driver == AGENT
        lw = 2.5 if is_agent else 0.8

        rect = Rectangle(
            (stint.start_lap, y - 0.42), width, 0.84,
            facecolor=color, edgecolor='black' if is_agent else edge,
            linewidth=lw, zorder=3 if is_agent else 2,
        )
        ax.add_patch(rect)

        if width > 4:
            ax.text(
                stint.start_lap + width / 2, y,
                f'C{stint.compound}', ha='center', va='center',
                fontsize=6, fontweight='bold', color='#333333',
            )

    # VSC/SC shading
    for lap in set(history.vsc_laps):
        ax.axvspan(lap - 0.5, lap + 0.5, alpha=0.25, color='#F1C40F', zorder=0)
    for lap in set(history.sc_laps):
        ax.axvspan(lap - 0.5, lap + 0.5, alpha=0.35, color='#E74C3C', zorder=0)

    ax.set_ylim(-0.5, len(sorted_drivers) - 0.5)
    ax.set_xlim(0, history.total_laps + 1)
    ax.set_yticks(range(len(sorted_drivers)))
    ax.set_yticklabels([
        f"P{final_positions[d]} {d.replace('Driver_', '')}"
        for d in sorted_drivers
    ], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Lap', fontsize=8)
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.set_title(
        f"Sim {sim_idx} (seed={seed}) | Total pits: {total_pits} | "
        f"SC: {sc_laps} laps, VSC: {vsc_laps} laps",
        fontsize=8, fontweight='bold',
    )


if __name__ == '__main__':
    print("Simulating 4 races...")
    histories = []
    for i, seed in enumerate(SEEDS):
        print(f"  Sim {i+1} (seed={seed})...")
        h = simulate_race_with_history(
            gp=GP, driver=AGENT, agent_strategy=AGENT_STRATEGY,
            deterministic=False, seed=seed, yf_enabled=True, verbose=False,
        )
        histories.append(h)
        final_pos = h.driver_positions[AGENT][-1]
        n_pits = sum(1 for s in h.stints if s.driver == AGENT and s.start_lap > 1)
        print(f"    Agent finished P{final_pos}, {n_pits} pit stop(s), "
              f"SC: {len(set(h.sc_laps))} laps, VSC: {len(set(h.vsc_laps))} laps")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"{GP} -- Rival Strategy Variability (4 Simulations)\n"
        f"Agent: {AGENT.replace('Driver_', '')} (C2->C1 1-stop) | "
        f"Rivals: Logit pit model + Conditional logit compound choice",
        fontsize=13, fontweight='bold', y=0.98,
    )

    for idx, (ax, h, seed) in enumerate(zip(axes.flat, histories, SEEDS)):
        plot_single_sim(ax, h, idx + 1, seed)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=COMPOUND_COLORS[c], edgecolor=COMPOUND_EDGE[c],
                       label=COMPOUND_LABELS[c])
        for c in [1, 2, 3]
    ]
    legend_elements.append(mpatches.Patch(facecolor='#F1C40F', alpha=0.25, label='VSC'))
    legend_elements.append(mpatches.Patch(facecolor='#E74C3C', alpha=0.35, label='SC'))
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                          linewidth=2.5, label=f'Agent ({AGENT.replace("Driver_", "")})'))

    fig.legend(
        handles=legend_elements, loc='lower center', ncol=6,
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    save_path = str(config.ROOT_DIR / 'bahrain_rival_stints_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.show()
