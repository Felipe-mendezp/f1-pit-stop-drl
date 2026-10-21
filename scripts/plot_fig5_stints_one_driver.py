"""
Figure 5 (paper-exact): tire strategies of the single-driver DRL agents vs
the QMIP optimum, reproducing Code/F1_one_driver/test.ipynb cells 17/18/24
(the original generator of stints_alone.pdf):

  1. Deterministic rollout of each one-driver agent per GP.
  2. QMIP plan obtained by MPC on the same deterministic environment (exact,
     since the env IS the QMIP generative model).
  3. Stint-reordering post-process: DRL stints are permuted to maximize the
     number of laps sharing a compound with the QMIP plan (stint order does
     not affect total time in the single-driver deterministic model).
  4. 3x3 grid, normalized laps, gold/red/white compound colors, bold labels
     'C# (len)', blue dashed lines at the QMIP pit laps.

Usage:
    python scripts/plot_fig5_stints_one_driver.py [--output figs/figure_5.png]
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import pickle
from itertools import permutations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import RecurrentPPO, TRPO

from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
from evaluate_qmip_vs_drl import build_qmip_params, qmip_choose_start_compound, qmip_choose_action
from model_loader import ModelLoader

# Same 9 GPs / order / colors as the original notebook (cell 24)
SELECTED_GP = {
    'Bahrain Grand Prix': ['C1', 'C2', 'C3'],
    'Belgian Grand Prix': ['C2', 'C3', 'C4'],
    'Dutch Grand Prix': ['C1', 'C2', 'C3'],
    'Emilia Romagna Grand Prix': ['C3', 'C4', 'C5'],
    'Hungarian Grand Prix': ['C3', 'C4', 'C5'],
    'Miami Grand Prix': ['C2', 'C3', 'C4'],
    'Saudi Arabian Grand Prix': ['C2', 'C3', 'C4'],
    'Singapore Grand Prix': ['C3', 'C4', 'C5'],
    'United States Grand Prix': ['C2', 'C3', 'C4'],
}
LAPS_DD = {
    'Bahrain Grand Prix': 57, 'Belgian Grand Prix': 44, 'Dutch Grand Prix': 72,
    'Emilia Romagna Grand Prix': 63, 'Hungarian Grand Prix': 70,
    'Miami Grand Prix': 57, 'Saudi Arabian Grand Prix': 50,
    'Singapore Grand Prix': 62, 'United States Grand Prix': 56,
}

MODELS = [('DQN', DQN), ('A2C', A2C), ('TRPO', TRPO), ('PPO', PPO),
          ('RecurrentPPO', RecurrentPPO)]
MODEL_DISPLAY = {'DQN': 'Double DQN', 'A2C': 'A2C', 'TRPO': 'TRPO',
                 'PPO': 'PPO', 'RecurrentPPO': 'Recurrent PPO'}

ONE_DRIVER_DIR = str(config.RL_AGENTS_DIR / 'one_driver')


def compound_colors_for(compounds):
    """Softest=red, middle=gold, hardest=white (per-GP role mapping)."""
    ordered = sorted(compounds, key=lambda c: int(c[1:]))
    return {ordered[2]: 'red', ordered[1]: 'gold', ordered[0]: 'white'}


def rollout(env, act_fn, seed=42):
    """Deterministic rollout; returns stint plan {start_lap_0idx: 'C#'}."""
    obs, _ = env.reset(seed=seed)
    plan = {}
    lap = 0
    done = False
    while not done:
        action = act_fn(obs, env)
        obs, _, done, _, _ = env.step(action)
        if env.lap_number == 1 and not plan:
            plan[0] = f'C{env.compound}'
        elif action > 0 and env.lap_number > 1:
            # stint change effective from current lap (0-indexed = lap_number-1)
            plan[env.lap_number - 1] = f'C{env.compound}'
    return plan


# ---- stint reordering post-process (notebook cell 18, verbatim logic) ----

def reorder_dict(dict_solution, reorder_indexes, laps):
    laps_per_stint = []
    starts = list(dict_solution.keys())
    for i in range(len(starts) - 1):
        laps_per_stint.append(starts[i + 1] - starts[i])
    laps_per_stint.append(laps - starts[-1])
    out = {}
    cur = 0
    for idx in reorder_indexes:
        out[cur] = dict_solution[starts[idx]]
        cur += laps_per_stint[idx]
    return out


def shared_laps(d1, d2, laps):
    s1, s2 = list(d1.keys()), list(d2.keys())
    e1 = s1[1:] + [laps]
    e2 = s2[1:] + [laps]
    total = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if d1[s1[i]] == d2[s2[j]]:
                lo, hi = max(s1[i], s2[j]), min(e1[i], e2[j])
                if lo < hi:
                    total += hi - lo
    return total


def best_reordering(mip_plan, drl_plan, laps):
    best, best_n = drl_plan, -1
    for perm in permutations(range(len(drl_plan))):
        cand = reorder_dict(drl_plan, perm, laps)
        n = shared_laps(mip_plan, cand, laps)
        if n > best_n:
            best_n, best = n, cand
    return best


def plan_to_triples(plan, laps):
    """{0:'C3', 19:'C3', 38:'C2'} -> [(1,19,'C3'), (20,38,'C3'), (39,laps,'C2')] (1-indexed)."""
    starts = sorted(plan.keys())
    triples = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else laps
        triples.append((s + 1, e, plan[s]))
    return triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='figs/figure_5.png')
    parser.add_argument('--models-dir', default=ONE_DRIVER_DIR)
    args = parser.parse_args()

    loader = ModelLoader()
    data_by_circuit = []
    optimal_plans = []

    for gp, compounds in SELECTED_GP.items():
        n_laps = LAPS_DD[gp]
        with open(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'), 'rb') as f:
            predictor = pickle.load(f)

        def make_env():
            return F1EnvOneDriverV2Optimized(
                n_laps=n_laps, available_compounds=compounds, gp=gp,
                driver='VER', model_laptime=predictor)

        # ---- QMIP plan via MPC on the deterministic env (exact optimum) ----
        qp = build_qmip_params(loader, gp)
        state = {'t_prev': [], 'started': False}

        def act_qmip(obs, env):
            if not state['started']:
                action, t0 = qmip_choose_start_compound(
                    qp['betas_dict'], qp['time_stop'], env.n_laps,
                    qp['qmip_to_compound'], qp['available_compounds'])
                state['t_prev'] = [t0]
                state['started'] = True
                return action
            t_cur = qp['compound_to_qmip'].get(env.compound, 'Medium')
            action = qmip_choose_action(
                qp['betas_dict'], qp['time_stop'], env.laps_left, t_cur,
                env.tyre_life, state['t_prev'], env.n_laps, 0,
                qp['qmip_to_compound'], qp['available_compounds'])
            if action > 0:
                name = qp['compound_to_qmip'].get(
                    qp['available_compounds'][action - 1], 'Medium')
                if name not in state['t_prev']:
                    state['t_prev'].append(name)
            return action

        qmip_plan = rollout(make_env(), act_qmip)
        optimal_plans.append(qmip_plan)
        print(f"{gp}: QMIP plan = {qmip_plan}")

        # ---- DRL agents (deterministic) ----
        gp_short = gp[:-11]
        circuit = []
        for name, cls in MODELS:
            path = os.path.join(args.models_dir, gp_short, name, 'best_model.zip')
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}")
                continue
            model = cls.load(path)
            lstm = (name == 'RecurrentPPO')
            mstate = {'lstm': None, 'start': np.ones((1,), dtype=bool)}

            def act_drl(obs, env, m=model, is_lstm=lstm, st=mstate):
                if is_lstm:
                    a, st['lstm'] = m.predict(obs, state=st['lstm'],
                                              episode_start=st['start'],
                                              deterministic=True)
                    st['start'] = np.zeros((1,), dtype=bool)
                else:
                    a, _ = m.predict(obs, deterministic=True)
                return int(np.asarray(a).ravel()[0])

            plan = rollout(make_env(), act_drl)
            plan = best_reordering(qmip_plan, plan, n_laps)  # visual alignment
            circuit.append({'GP': gp, 'model': MODEL_DISPLAY[name], 'stint': plan})
            print(f"  {MODEL_DISPLAY[name]:14s}: {plan}")
        data_by_circuit.append(circuit)

    # ---- plotting (notebook cell 24, verbatim style) ----
    gp_names = list(SELECTED_GP.keys())
    # 2026-10-21: F1_FIG5_GRID=2x4 gives the 2x4 layout of the manuscript (default 3x3, as published)
    nrows, ncols = (2, 4) if os.environ.get('F1_FIG5_GRID') == '2x4' else (3, 3)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5) if ncols == 4 else (15, 7), sharex=True,
                            constrained_layout=True)

    for ci, ax in enumerate(axs.flat):
        circuit_data = data_by_circuit[ci]
        gp_name = gp_names[ci]
        total = LAPS_DD[gp_name]
        colors = compound_colors_for(SELECTED_GP[gp_name])
        qmip_plan = optimal_plans[ci]

        for i, d in enumerate(circuit_data):
            for start_lap, end_lap, comp in plan_to_triples(d['stint'], total):
                s = (start_lap - 1) / total
                e = (end_lap - 1) / total
                color = colors[comp]
                ax.broken_barh([(s, e - s + 1 / total)], (i * 10, 9),
                               facecolors=color, edgecolors='black')
                ax.text((s + e) / 2, i * 10 + 4.5, f'{comp} ({end_lap - start_lap + 1})',
                        ha='center', va='center',
                        color='white' if color == 'red' else 'black',
                        fontweight='bold')

        # QMIP optimal pit laps as vertical dashed lines
        for lap in sorted(k for k in qmip_plan if k != 0):
            ax.axvline(x=lap / total, color='blue', linestyle='--')

        # QMIP optimal stint row on top
        qrow = len(circuit_data)
        for start_lap, end_lap, comp in plan_to_triples(qmip_plan, total):
            s = (start_lap - 1) / total
            e = (end_lap - 1) / total
            color = colors[comp]
            ax.broken_barh([(s, e - s + 1 / total)], (qrow * 10, 9),
                           facecolors=color, edgecolors='black')
            ax.text((s + e) / 2, qrow * 10 + 4.5, f'{comp} ({end_lap - start_lap + 1})',
                    ha='center', va='center',
                    color='white' if color == 'red' else 'black', fontweight='bold')

        ax.set_title(gp_name, color='black', fontweight='bold')
        ax.set_ylim(-1, 10 * (len(circuit_data) + 1))
        ax.set_xlim(0, 1)
        if ci % ncols == 0:
            ax.set_ylabel('Model')
            ax.set_yticks([i * 10 + 4.5 for i in range(len(circuit_data))] + [qrow * 10 + 4.5])
            ax.set_yticklabels([d['model'] for d in circuit_data] + ['QMIP Optimal'], color='black')
        else:
            ax.set_yticks([i * 10 + 4.5 for i in range(len(circuit_data))] + [qrow * 10 + 4.5])
            ax.set_yticklabels([])
        if ci >= (nrows - 1) * ncols:
            ax.set_xlabel('Normalized Laps')

    qmip_line = Line2D([0], [0], color='blue', linestyle='--', label='QMIP Optimal')
    hard_patch = mpatches.Patch(facecolor='white', label='Hard', edgecolor='black')
    medium_patch = mpatches.Patch(facecolor='gold', label='Medium', edgecolor='black')
    soft_patch = mpatches.Patch(facecolor='red', label='Soft', edgecolor='black')
    fig.legend(handles=[qmip_line, hard_patch, medium_patch, soft_patch],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13 if ncols == 4 else -0.06), frameon=True)

    # 2026-10-21: manuscript fonts (opt-in F1_PAPER_FONTS=1, see paper_fonts_20261021.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paper_fonts_20261021; paper_fonts_20261021.apply(fig)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    fig.savefig(args.output.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved paper-exact Figure 5 to {args.output}")


if __name__ == '__main__':
    main()
