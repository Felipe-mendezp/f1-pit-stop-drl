import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Counterfactual rollouts for the Decision Support Visualization (ANOR paper, Sec.
"Decision Support Visualization for Human-AI Collaboration", Algorithm 1).

For a fixed race, lap n and driver, force each candidate action
a in {No-pit, Hard, Medium, Soft} at lap n, then play the remaining laps with the
trained Recurrent PPO policy acting DETERMINISTICALLY (argmax), i.e. exactly what
the agent would really do after the forced action — the same policy whose
recommendation the figure highlights and the same continuation the stint diagram
draws. Each rollout draws independent realizations of the stochastic race events
(yellow flags, lap-time noise, overtakes) AND of the rivals' pit decisions (full
re-sample variant). The final outcome (position, total time, points) is recorded,
yielding an empirical distribution of outcomes conditional on the action taken at
lap n.

Pass --stochastic-cont to sample the continuation from the policy instead (the
pre-2026-07-25 behaviour, kept as a sensitivity variant). Beware: under that
protocol the "No pit" arm means "postpone and choose freely later", which is a
different policy from the one being recommended and systematically favours
staying out.

Outputs a tidy CSV + a meta JSON to results/decision_support/. Plotting is done
separately by plot_decision_support.py so the figure can be restyled without
re-running the (expensive) simulations.

Usage:
    python scripts/decision_support_rollouts.py --lap 15 --M 500
"""

import copy
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import torch

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader

# Reuse the GP configuration and helpers from the main evaluation script so the
# scenario (rival starting compounds, 2024 grid) matches the rest of the paper.
from evaluate_all_gps import (
    load_gp_config,
    build_rival_strategies,
    GP_CONFIGS,
    ALGO_CLASSES,
    NORM_OBS_KEYS,
)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# F1 points for top-10 (championship points outcome).
POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# Heavy, read-only model attributes detached before deepcopy so snapshot/restore
# is cheap and does not duplicate the large predictor / logit objects.
HEAVY_MODEL_ATTRS = [
    'fast_predictor_clear',
    'fast_predictor_yf',
    'logit_overtake',
    'rival_pitstop_logit',
    'rival_compound_logit',
]


def batch_obs(obs):
    """Wrap a single raw_env observation dict into the (n_envs=1) batched form
    that VecNormalize / model.predict expect."""
    return {k: np.array([v]) for k, v in obs.items()}


def snapshot_env(env):
    """Deep-copy the mutable race state of `env`, detaching the heavy read-only
    models first (and restoring them afterwards on both original and copy)."""
    detached = {attr: getattr(env, attr) for attr in HEAVY_MODEL_ATTRS}
    for attr in HEAVY_MODEL_ATTRS:
        setattr(env, attr, None)
    try:
        snap = copy.deepcopy(env)
    finally:
        for attr, val in detached.items():
            setattr(env, attr, val)
            setattr(snap, attr, val)  # share the same model objects (read-only)
    return snap


def restore_from(snap):
    """Produce a fresh independent env from a snapshot (models shared)."""
    detached = {attr: getattr(snap, attr) for attr in HEAVY_MODEL_ATTRS}
    for attr in HEAVY_MODEL_ATTRS:
        setattr(snap, attr, None)
    try:
        env = copy.deepcopy(snap)
    finally:
        for attr, val in detached.items():
            setattr(snap, attr, val)
            setattr(env, attr, val)
    return env


def action_labels(available_compounds):
    """Map action index -> label. Action 0 = No pit; 1..n -> compounds, ordered
    hardest -> softest (available_compounds is sorted ascending = C-number,
    lower C = harder). For Bahrain [1,2,3] -> Hard/Medium/Soft."""
    comps = [c if isinstance(c, int) else int(c[1:]) for c in available_compounds]
    n = len(comps)
    # Relative hardness names used in F1 weekend nomenclature.
    rel = {3: ['Hard', 'Medium', 'Soft'],
           2: ['Hard', 'Soft'],
           4: ['Hard', 'Medium', 'Soft', 'Hyper']}
    names = rel.get(n, [f'C{c}' for c in comps])
    labels = {0: 'No pit'}
    for i, c in enumerate(comps):
        labels[i + 1] = f'{names[i]} (C{c})'
    return labels


def run_prefix_multi(model, vec_env, raw_env, laps, initial_compounds, prefix_seed):
    """Step the policy deterministically from reset, snapshotting the race state
    at each requested lap in `laps`. Runs the prefix only ONCE (to the largest
    target lap). Returns dict lap -> {obs, lstm, rec_action, snap, agent_state}."""
    targets = sorted(set(int(n) for n in laps))
    np.random.seed(prefix_seed)
    torch.manual_seed(prefix_seed)
    raw_env.np_random = np.random.default_rng(prefix_seed)
    raw_env.action_space.seed(prefix_seed)

    # FIX (2026-07-21, audit M2): apply the real 2024 starting compounds
    # BEFORE reset (reset() reads eval_initial_compounds when building the
    # rival Drivers, independent of eval mode). Previously the compounds were
    # overwritten AFTER reset, so the lap-0 observation — on which the agent
    # conditions its own initial-compound choice — showed randomly sampled
    # rival compounds instead of the real grid.
    raw_env.eval_initial_compounds = dict(initial_compounds or {})

    obs, _ = raw_env.reset(seed=prefix_seed)

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    out = {}
    history = []  # (lap, agent_compound_during_lap, pitted_this_lap) for the prefix
    yf_laps = []  # (lap, 'SC'|'VSC') realized in the common deterministic prefix

    def capture(cur_obs, cur_lstm):
        norm = vec_env.normalize_obs(batch_obs(cur_obs))
        rec_action, lstm_after = model.predict(
            norm, state=cur_lstm, episode_start=np.zeros((1,), dtype=bool),
            deterministic=True)
        return {
            'obs': cur_obs,
            'lstm': lstm_after,
            'rec_action': int(np.asarray(rec_action).squeeze()),
            'snap': snapshot_env(raw_env),
            'agent_state': (int(raw_env.agent_driver.position),
                            int(raw_env.agent_driver.compound),
                            int(raw_env.agent_driver.tyrelife)),
            'history': list(history),
            'yf_laps': list(yf_laps),
        }

    max_lap = max(targets)
    guard = 0
    while raw_env.lap_number <= max_lap:
        if raw_env.lap_number in targets:
            out[raw_env.lap_number] = capture(obs, lstm_states)
        if raw_env.lap_number == max_lap:
            break
        norm = vec_env.normalize_obs(batch_obs(obs))
        action, lstm_states = model.predict(
            norm, state=lstm_states, episode_start=episode_start, deterministic=True)
        episode_start = np.zeros((1,), dtype=bool)
        cur_lap = raw_env.lap_number
        act = int(np.asarray(action).squeeze())
        obs, _, done, _, _ = raw_env.step(act)
        # Pit EJECUTADO, no emitido (2026-09-03): bajo sc_bunching el env
        # convierte en no-op los pits en vueltas SC 2+, y con rule_mask los
        # redirige; prev_action refleja la acción efectiva del agente en esta
        # vuelta. La vuelta 0 es la elección de compuesto (act != 0).
        executed = (act != 0) if cur_lap == 0 else (int(raw_env.agent_driver.prev_action) != 0)
        history.append((cur_lap, int(raw_env.agent_driver.compound), executed))
        # Yellow flags realized in the COMMON prefix. In the fully-simulated
        # variant these were invisible: nothing recorded them, so the evolution
        # figure could not shade them and a scenario believed to be SC/VSC-free
        # went unnoticed (Dutch prefix seed 34 actually has 5 yellow-flag laps).
        if raw_env.sc > 0:
            yf_laps.append((cur_lap, 'SC'))
        elif raw_env.vsc > 0:
            yf_laps.append((cur_lap, 'VSC'))
        guard += 1
        if done or guard > raw_env.n_laps + 5:
            raise RuntimeError(f"Race ended before reaching lap {max_lap} "
                               f"(lap_number={raw_env.lap_number}).")

    missing = [n for n in targets if n not in out]
    if missing:
        raise RuntimeError(f"Could not capture laps {missing} (n_laps={raw_env.n_laps}).")
    return out


def _rule_infeasible(env, act):
    """True iff `act` is not part of any rule-feasible strategy given the current
    state, under the mandatory two-compound rule. Two cases:

    (a) a pit onto the compound currently fitted while the rule is still
        unsatisfied: it pays the full pit loss without progressing toward the
        mandatory change; and
    (b) NOT pitting on the last lap on which the mandatory change could still be
        made (rule unsatisfied and lap_number >= n_laps - 1): coasting to the
        flag here guarantees disqualification, so skipping the stop is itself
        infeasible and the agent must take a legal different-compound stop.

    Masking (b) as well as (a) makes the continuation exactly rule-feasible, so a
    forced same-compound stop is realised as a costly EXTRA stop (a legal change
    is still made before the flag) rather than a disqualification."""
    changed = env.agent_driver.has_changed_compound
    if act == 0:
        return (not changed) and env.lap_number >= env.n_laps - 1
    if changed:
        return False
    return env.available_compounds[act - 1] == env.agent_driver.compound


def switch_future_to_stochastic(env):
    """Protocolo sim-future (2026-08-13): condicionar en el PASADO real,
    simular el FUTURO. Se llama sobre la copia restaurada del snapshot de la
    vuelta de decisión X: desde X (inclusive) los SC/VSC salen del modelo de
    hazard y los rivales del logit, porque en X ese futuro es incognoscible.

    Mecánica del traspaso:
    - eval=False: _update_sc_vsc pasa a muestrear y _simulate_rival_pits al
      logit; _update_safety_car_duration vuelve a decrementar contadores.
    - Se vacían laps_sc/laps_vsc/rival_actions_dict (cinturón y tirantes).
    - Los contadores sc/vsc/laps_left_* del snapshot NO se tocan: un SC/VSC
      ACTIVO al llegar a X es información presente conocida y termina su
      duración real de forma natural (en modo muestreado solo se sortean
      eventos nuevos cuando no hay uno activo); después manda el hazard.
    - Los DNFs ya ocurridos persisten (Driver.is_dnf); no se generan nuevos
      (solo el camino de acciones reales los programa)."""
    env.eval = False
    env.laps_sc = []
    env.laps_vsc = []
    env.rival_actions_dict = {}


def rollout_forced(model, vec_env, snap, obs_n, lstm_after_n, forced_action, seed,
                   legal_mask=False, det_cont=True, sim_future=False):
    """One Monte Carlo rollout: force `forced_action` at lap n on a fresh copy of
    the snapshot, then continue with the policy. Returns
    (final_position, total_time, points, n_stops, pit_laps, masked_hits).

    det_cont=True (DEFAULT): the policy acts DETERMINISTICALLY (argmax) after the
    forced action, i.e. exactly what the DRL would really do; the spread then
    reflects only environment stochasticity (lap-time noise, and stochastic
    rivals when they are not pinned to the real race). This is the estimand the
    figure claims to show: Q^pi(s_n, a) for the SAME deterministic policy whose
    recommendation is highlighted, and it matches the continuation drawn by the
    stint diagram (plot_decision_stints.py).
    det_cont=False: continuation actions are SAMPLED from the policy, so the arm
    "No pit" is really "postpone, then choose freely later" — a different policy
    from the one being recommended, which systematically favours staying out.
    Kept only as a sensitivity variant (--stochastic-cont).

    If legal_mask is True, continuation actions are RESTRICTED to rule-feasible
    ones: a same-compound pit while the two-compound rule is unsatisfied, or
    skipping the mandatory stop at the last feasible lap, is masked out (via
    rejection sampling when stochastic, or a direct redirect when deterministic).
    The LSTM state update depends only on the observation, not the action, so
    rejection sampling is exact."""
    env = restore_from(snap)
    env.np_random = np.random.default_rng(seed)
    if sim_future:
        switch_future_to_stochastic(env)
    # Seed the torch RNG too: with a stochastic continuation the sampled actions
    # would otherwise depend on the global RNG state, i.e. on how many predict()
    # calls happened before (action order, lap chunking) -> not reproducible and
    # not paired across arms. No-op under det_cont, kept for both paths.
    torch.manual_seed(seed)

    lstm = copy.deepcopy(lstm_after_n)
    episode_start = np.zeros((1,), dtype=bool)

    n_stops = 1 if int(forced_action) != 0 else 0
    pit_laps = [int(env.lap_number)] if int(forced_action) != 0 else []
    masked_hits = 0

    # Forced action at lap n (never masked: it is the counterfactual under study).
    obs, _, done, _, info = env.step(int(forced_action))

    while not done:
        norm = vec_env.normalize_obs(batch_obs(obs))
        action, new_lstm = model.predict(
            norm, state=lstm, episode_start=episode_start, deterministic=det_cont)
        act = int(np.asarray(action).squeeze())
        if legal_mask and _rule_infeasible(env, act):
            masked_hits += 1
            if det_cont:
                # Deterministic policy always returns the same action, so go
                # straight to the rule-feasible redirect (no rejection sampling).
                act = next(a for a in range(1, env._n_compounds + 1)
                           if env.available_compounds[a - 1] != env.agent_driver.compound)
            else:
                # Rejection-sample from the masked distribution (same obs/state).
                for _ in range(50):
                    action, new_lstm = model.predict(
                        norm, state=lstm, episode_start=episode_start, deterministic=False)
                    act = int(np.asarray(action).squeeze())
                    if not _rule_infeasible(env, act):
                        break
                else:
                    # Degenerate distribution: redirect to a different compound.
                    act = next(a for a in range(1, env._n_compounds + 1)
                               if env.available_compounds[a - 1] != env.agent_driver.compound)
        lstm = new_lstm
        if act != 0:
            n_stops += 1
            pit_laps.append(int(env.lap_number))
        obs, _, done, _, info = env.step(act)

    changed = info.get('has_changed_compound', False)
    raw_pos = info.get('final_position', env.agent_driver.position)
    # DSQ (never changed compound) -> last place P20 (matches evaluate_all_gps).
    pos = 20 if not changed else int(raw_pos)
    total_time = float(info.get('total_time', env.agent_driver.total_time))
    points = POINTS_SYSTEM[pos - 1] if (pos <= 10 and changed) else 0
    return pos, total_time, points, n_stops, pit_laps, masked_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp', default='Bahrain Grand Prix')
    parser.add_argument('--driver', default='Driver_ALO')
    parser.add_argument('--model-dir', default='Bahrain_ALO')
    parser.add_argument('--laps', type=int, nargs='+', default=[15],
                        help='Decision lap(s) n; multiple laps share one prefix run')
    parser.add_argument('--M', type=int, default=500, help='Rollouts per action')
    parser.add_argument('--prefix-seed', type=int, default=0)
    parser.add_argument('--base-seed', type=int, default=1000)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--legal-mask', action='store_true',
                        help='Restrict continuation actions to rule-feasible ones '
                             '(mask same-compound pit while two-compound rule unmet)')
    parser.add_argument('--real-2024', action='store_true',
                        help='Condition on the actual 2024 race: fix SC/VSC, rivals '
                             'and initial compounds to the real event (evaluation '
                             'mode). Rollouts then vary only by lap-time noise and '
                             'the agent policy, all sharing the real race context.')
    parser.add_argument('--real-yf', action='store_true',
                        help='Fix ONLY the SC/VSC events to the real 2024 race; '
                             'everything else (rival pit decisions, lap-time noise) '
                             'is simulated/stochastic. Hybrid of the seed and '
                             'real-2024 variants.')
    parser.add_argument('--stochastic-cont', action='store_true',
                        help='SAMPLE the continuation from the policy instead of '
                             'acting deterministically (pre-2026-07-25 default). '
                             'Sensitivity variant only: it evaluates a different '
                             'policy from the one being recommended.')
    parser.add_argument('--sim-future', action='store_true',
                        help='Protocolo 2026-08-13: el prefijo 1..X-1 conserva el '
                             'condicionamiento de la variante (--real-2024 o '
                             '--real-yf), pero las continuaciones desde la vuelta '
                             'de decisión X son 100%% SIMULADAS (SC/VSC por hazard, '
                             'rivales por logit) — en X el futuro es incognoscible. '
                             'Un SC/VSC activo al llegar a X termina su duración '
                             'real y luego manda el hazard.')
    parser.add_argument('--algo', default=os.environ.get('F1_DS_ALGO', 'RecurrentPPO'),
                        choices=list(ALGO_CLASSES.keys()),
                        help='Agente del decision support (2026-09-03): subcarpeta '
                             '<RL_AGENTS_DIR>/<model_dir>/<algo>/best_model.zip. '
                             'Default RecurrentPPO (o F1_DS_ALGO).')
    parser.add_argument('--tag-suffix', default='',
                        help='Suffix appended to the output tag (e.g. "_masked")')
    parser.add_argument('--fast-env', action='store_true',
                        help='Numpy fast path for rival/overtake logits '
                             '(validated against statsmodels at init, ~6x '
                             'faster env steps).')
    # Env-physics fixes for the pit-timing artefacts (2026-08-04), both opt-in
    # so the default run reproduces the legacy bundles bit-for-bit:
    parser.add_argument('--pit-det-pass', action='store_true',
                        help='Fix (a): resolve overtakes involving a driver on '
                             'an in-/out-lap deterministically (pred_gap < 0) '
                             'instead of extrapolating the overtake logit, so a '
                             'car in the pit lane cannot weld the train behind '
                             'it ("free pit stop" artefact).')
    parser.add_argument('--following-gap-scale', type=float, default=0.0,
                        help='Fix (b): denied overtakes settle at MIN_GAP + '
                             'Exponential(scale) instead of the constant 0.4s '
                             'that welds the field into trains. Calibrated on '
                             'data/session_2024_V2.csv Interval_front in the '
                             'following regime (0.4-2.5s]: median 1.10s, mean '
                             'excess 0.87s => scale 1.0. 0.0 = legacy.')
    args = parser.parse_args()

    # Deterministic continuation is the default; --stochastic-cont opts out.
    det_cont = not args.stochastic_cont

    # Default the tag suffix by variant (unless overridden).
    if not args.tag_suffix:
        if args.real_2024:
            args.tag_suffix = '_real'
        elif args.real_yf:
            args.tag_suffix = '_realyf'

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'decision_support')
    os.makedirs(out_dir, exist_ok=True)

    # --- Find the matching GP_CONFIGS entry for rival starting compounds / grid ---
    entry = next((e for e in GP_CONFIGS
                  if e['gp_name'] == args.gp and e['agent_driver'] == args.driver), None)
    if entry is None:
        raise SystemExit(f"No GP_CONFIGS entry for {args.gp} / {args.driver}")

    gp_config = load_gp_config(args.gp)
    rival_actions, initial_compounds = build_rival_strategies(entry)
    initial_positions = gp_config['initial_positions_list_2024']

    # --- Build env ---
    # Default (seed) mode: full re-sample, stochastic rivals + SC/VSC.
    # --real-2024 mode: evaluation mode with the actual 2024 SC/VSC laps and rival
    # pit strategies, so every rollout shares the real race context and varies only
    # by lap-time noise and the agent's own (stochastic) continuation.
    loader = ModelLoader()
    raw_env = create_f1_env_all_drivers(
        gp=args.gp, driver=args.driver,
        deterministic=False, yf_enabled=True,
        initial_positions=initial_positions, loader=loader, verbose=False,
        fast_rivals=args.fast_env)

    # Opt-in physics fixes (survive snapshot/restore: plain attrs, deep-copied).
    raw_env.pit_pass_deterministic = args.pit_det_pass
    raw_env.following_gap_scale = args.following_gap_scale
    if args.pit_det_pass or args.following_gap_scale:
        print(f"Env physics fixes: pit_det_pass={args.pit_det_pass} "
              f"following_gap_scale={args.following_gap_scale}")

    if args.real_2024:
        raw_env.set_evaluation_mode(
            eval_mode=True,
            laps_vsc=gp_config['laps_vsc_2024'],
            laps_sc=gp_config['laps_sc_2024'],
            initial_positions=initial_positions,
            rival_actions=rival_actions,
            rival_initial_compounds=initial_compounds)
        print(f"REAL-2024 mode: SC laps={gp_config['laps_sc_2024']} "
              f"VSC laps={gp_config['laps_vsc_2024']} "
              f"({len(rival_actions)} rivals fixed to real strategies)")
    elif args.real_yf:
        # Fix ONLY the SC/VSC timeline to the real race; rivals stochastic
        # (empty rival_actions => logit model path), lap times stochastic.
        raw_env.set_evaluation_mode(
            eval_mode=True,
            laps_vsc=gp_config['laps_vsc_2024'],
            laps_sc=gp_config['laps_sc_2024'],
            initial_positions=initial_positions,
            rival_actions={})
        print(f"REAL-YF mode: SC laps={gp_config['laps_sc_2024']} "
              f"VSC laps={gp_config['laps_vsc_2024']} fixed; "
              f"rivals + lap times stochastic")

    vec_env = DummyVecEnv([lambda: raw_env])
    model_path = os.path.join(str(config.RL_AGENTS_DIR), args.model_dir,
                              args.algo, 'best_model.zip')
    vecnorm_path = os.path.join(os.path.dirname(model_path), 'vecnormalize_stats.pkl')
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS,
                               clip_reward=10000.0)
    vec_env.training = False
    vec_env.norm_reward = False

    model = ALGO_CLASSES[args.algo].load(model_path, env=vec_env)
    model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(
        timespec='seconds')
    print(f"Loaded model type: {type(model).__name__} from {model_path} "
          f"(mtime {model_mtime})")
    print(f"Continuation: {'DETERMINISTIC (argmax)' if det_cont else 'SAMPLED (stochastic)'}")

    labels = action_labels(raw_env.available_compounds)
    print(f"Action labels: {labels}")

    # --- Single prefix run, snapshot at each requested lap ---
    captures = run_prefix_multi(
        model, vec_env, raw_env, args.laps, initial_compounds, args.prefix_seed)

    candidate_actions = sorted(labels.keys())  # [0, 1, 2, 3]
    for n in sorted(captures.keys()):
        cap = captures[n]
        pos0, comp0, life0 = cap['agent_state']
        print(f"\n=== Lap {n} === Agent P{pos0}, compound C{comp0}, tyre life {life0}. "
              f"DRL rec: action {cap['rec_action']} = {labels[cap['rec_action']]}")

        rows = []
        for a in candidate_actions:
            positions, stops = [], []
            for m in range(args.M):
                pos, ttime, pts, nstops, pit_laps, mhits = rollout_forced(
                    model, vec_env, cap['snap'], cap['obs'], cap['lstm'],
                    a, args.base_seed + m, legal_mask=args.legal_mask,
                    det_cont=det_cont, sim_future=args.sim_future)
                rows.append({'action': a, 'action_label': labels[a], 'rollout': m,
                             'final_position': pos, 'total_time': ttime, 'points': pts,
                             'n_stops': nstops,
                             'pit_laps': ';'.join(str(x) for x in pit_laps),
                             'masked_hits': mhits})
                positions.append(pos)
                stops.append(nstops)
            positions = np.array(positions)
            print(f"  action {a} ({labels[a]:14s}): "
                  f"median P{np.median(positions):.0f}  "
                  f"mean P{positions.mean():.2f}  "
                  f"P(top10)={np.mean(positions <= 10):.2f}  "
                  f"stops={np.mean(stops):.2f}")

        tag = f"{args.model_dir.lower()}{args.tag_suffix}_lap{n}"
        csv_path = os.path.join(out_dir, f"{tag}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['action', 'action_label', 'rollout',
                                                   'final_position', 'total_time', 'points',
                                                   'n_stops', 'pit_laps', 'masked_hits'])
            writer.writeheader()
            writer.writerows(rows)

        meta = {
            'gp': args.gp, 'driver': args.driver, 'lap': n, 'M': args.M,
            'n_laps': raw_env.n_laps,
            # Deterministic prefix trace up to this lap: (lap, compound, pitted).
            # Lets derive_real_picks.py recover the prefix pit laps without
            # re-running anything.
            'history': [[int(l), int(c), bool(p)] for l, c, p in cap['history']],
            # Yellow flags actually realized in the common prefix, so the figures
            # can shade them in every variant (not only when they come pinned
            # from the real 2024 race).
            'prefix_yf_laps': [[int(l), str(k)] for l, k in cap.get('yf_laps', [])],
            'available_compounds': [int(c) for c in raw_env.available_compounds],
            'action_labels': labels,
            'recommended_action': cap['rec_action'],
            'recommended_label': labels[cap['rec_action']],
            'agent_start_position': gp_config['initial_positions_2024'].get(args.driver),
            'agent_state_at_lap': {'position': pos0, 'compound': comp0, 'tyrelife': life0},
            'prefix_seed': args.prefix_seed, 'base_seed': args.base_seed,
            'variant': (('real_2024' if args.real_2024 else
                         'real_yf' if args.real_yf else 'full_resample')
                        + ('_masked' if args.legal_mask else '')
                        + ('' if det_cont else '_stochcont')),
            'legal_mask': args.legal_mask,
            # Continuation protocol + which agent set produced this (results are
            # silently model-set dependent: with the default rl_agents (v1,
            # pre-2026-07-21-fix models) the policy is degenerate).
            'det_cont': det_cont,
            # Protocolo sim-future (2026-08-13): pasado real / futuro simulado.
            'sim_future': args.sim_future,
            # Env-physics fixes (2026-08-04); both False/0.0 in legacy runs.
            'pit_pass_deterministic': args.pit_det_pass,
            'following_gap_scale': args.following_gap_scale,
            # SC bunching (2026-08-29): llega por env vars de la factory
            # (F1_SC_BUNCHING / F1_SC_BUNCH_DELTA); se registra del env real.
            'sc_bunching': bool(getattr(raw_env, 'sc_bunching', False)),
            'sc_bunch_delta': float(getattr(raw_env, 'sc_bunch_delta', 0.0) or 0.0),
            # Reward mix (2026-09-02): penalización de DSQ efectiva (F1_DSQ_PENALTY).
            'dsq_penalty': float(getattr(raw_env, '_penalty_disqualification', float('nan'))),
            # Cierre del loophole del pit en la última vuelta (F1_LASTLAP_PITOUT, 2026-09-03).
            'last_lap_pit_full_cost': bool(getattr(raw_env, 'last_lap_pit_full_cost', False)),
            # Reward mix-v2 relativo al pelotón (F1_REWARD_RELATIVE, 2026-09-05); solo procedencia.
            'reward_relative': bool(getattr(raw_env, 'reward_relative', False)),
            # Simulation model tree (2026-08-07): results depend on which
            # overtake logit is loaded (F1_SIMULATION_DIR switches trees).
            'simulation_dir': str(config.SIMULATION_DIR),
            'rl_agents_dir': str(config.RL_AGENTS_DIR),
            'algo': args.algo,
            'model_path': model_path,
            'model_mtime': model_mtime,
            'real_2024': args.real_2024,
            'real_yf': args.real_yf,
            'laps_sc_2024': gp_config['laps_sc_2024'] if (args.real_2024 or args.real_yf) else None,
            'laps_vsc_2024': gp_config['laps_vsc_2024'] if (args.real_2024 or args.real_yf) else None,
        }
        with open(os.path.join(out_dir, f"{tag}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2, default=int)
        print(f"  Wrote {len(rows)} rows -> {csv_path}")


if __name__ == '__main__':
    main()
