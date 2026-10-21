"""Screening of single-driver (Fig. 5 / Table N.1) checkpoints — 2026-09-23.

For every paper GP and algorithm found under <models-dir>/<GP short>/<algo>/
best_model.zip, roll the deterministic policy out in the deterministic
single-driver environment and report:

  stops    number of pit stops (actions > 0 after lap 0)
  changed  whether the mandatory two-compound rule is satisfied
  gap      total race time minus the QMIP optimum (seconds; same statistic
           as Table N.1 of the supplementary / table_M1 here)
  sane     changed AND 1 <= stops <= 4   (same bound as the multi-driver
           prefix-pit screening: a degenerate agent is one that never pits,
           never changes compound, or pit-spams)

Writes a TSV with one row per (GP, algo) and, with --degenerate-out, the list
of pairs that must go through the rescue-seed step.

Usage:
    F1_RL_AGENTS_DIR=<tree> python scripts/screen_one_driver_20260923.py \
        --out <tsv> --degenerate-out <tsv> [--pairs <tsv of GP\talgo to screen>]
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import pickle

import numpy as np

from plot_fig5_stints_one_driver import SELECTED_GP, LAPS_DD, MODELS, MODEL_DISPLAY
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
from evaluate_qmip_vs_drl import (build_qmip_params, qmip_choose_start_compound,
                                  qmip_choose_action, qmip_set_ages)
from model_loader import ModelLoader


def rollout_full(env, act_fn, seed=42):
    """Deterministic rollout -> (total_time, changed_compound, n_stops, plan)."""
    obs, _ = env.reset(seed=seed)
    done = False
    stops = 0
    plan = []
    while not done:
        a = act_fn(obs, env)
        lap_before = env.lap_number
        obs, _, done, _, _ = env.step(a)
        if lap_before == 0:
            plan.append(f"start C{env.compound}")
        elif a > 0:
            stops += 1
            plan.append(f"L{lap_before}->C{env.compound}")
    return float(env.total_time), bool(env.change_compound), stops, ' '.join(plan)


def qmip_actor(qp):
    state = {'t_prev': [], 'started': False}

    def act(obs, env):
        if not state['started']:
            a, t0 = qmip_choose_start_compound(
                qp['betas_dict'], qp['time_stop'], env.n_laps,
                qp['qmip_to_compound'], qp['available_compounds'],
                set_ages=qmip_set_ages(env, qp['qmip_to_compound']))
            state['t_prev'] = [t0]; state['started'] = True
            return a
        t_cur = qp['compound_to_qmip'].get(env.compound, 'Medium')
        a = qmip_choose_action(
            qp['betas_dict'], qp['time_stop'], env.laps_left, t_cur,
            env.tyre_life, state['t_prev'], env.n_laps, 0,
            qp['qmip_to_compound'], qp['available_compounds'],
            set_ages=qmip_set_ages(env, qp['qmip_to_compound']))
        if a > 0:
            nm = qp['compound_to_qmip'].get(qp['available_compounds'][a - 1], 'Medium')
            if nm not in state['t_prev']:
                state['t_prev'].append(nm)
        return a
    return act


def drl_actor(model, lstm):
    st = {'lstm': None, 'start': np.ones((1,), dtype=bool)}

    def act(obs, env):
        if lstm:
            a, st['lstm'] = model.predict(obs, state=st['lstm'],
                                          episode_start=st['start'], deterministic=True)
            st['start'] = np.zeros((1,), dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=True)
        return int(np.asarray(a).ravel()[0])
    return act


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir', default=str(config.RL_AGENTS_DIR / 'one_driver'))
    ap.add_argument('--out', required=True, help='TSV with one row per (GP, algo)')
    ap.add_argument('--degenerate-out', default=None,
                    help='TSV (GP<TAB>algo) of the pairs that are NOT sane')
    ap.add_argument('--pairs', default=None,
                    help='TSV (GP<TAB>algo) restricting the screening to these pairs')
    ap.add_argument('--max-stops', type=int, default=4,
                    help='Sane if 1 <= stops <= max-stops (default 4, same bound as the '
                         'multi-driver prefix-pit screening; Belgian optimum has 3 stops).')
    args = ap.parse_args()

    pairs = None
    if args.pairs:
        pairs = set()
        for line in open(args.pairs):
            line = line.rstrip('\n')
            if line and not line.startswith('#'):
                gp, algo = line.split('\t')[:2]
                pairs.add((gp, algo))

    loader = ModelLoader()
    rows = []
    degenerate = []
    for gp, compounds in SELECTED_GP.items():
        n_laps = LAPS_DD[gp]
        with open(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'), 'rb') as f:
            predictor = pickle.load(f)

        def make_env():
            return F1EnvOneDriverV2Optimized(
                n_laps=n_laps, available_compounds=compounds, gp=gp,
                driver='VER', model_laptime=predictor)

        wanted = [(n, c) for n, c in MODELS if pairs is None or (gp, n) in pairs]
        if not wanted:
            continue
        qp = build_qmip_params(loader, gp)
        t_opt, _, s_opt, plan_opt = rollout_full(make_env(), qmip_actor(qp))
        print(f"\n{gp}: QMIP optimum {t_opt:.2f} s, {s_opt} stops  [{plan_opt}]")

        for name, cls in wanted:
            path = os.path.join(args.models_dir, gp[:-11], name, 'best_model.zip')
            if not os.path.exists(path):
                print(f"  {MODEL_DISPLAY[name]:14s} MISSING {path}")
                rows.append((gp, name, 'nan', 'nan', 'nan', 'MISSING', ''))
                degenerate.append((gp, name))
                continue
            model = cls.load(path)
            t, changed, stops, plan = rollout_full(make_env(), drl_actor(model, name == 'RecurrentPPO'))
            gap = t - t_opt
            sane = changed and 1 <= stops <= args.max_stops
            flag = 'sane' if sane else ('NO-CHANGE' if not changed else 'PIT-SPAM')
            print(f"  {MODEL_DISPLAY[name]:14s} gap={gap:8.3f} s  stops={stops}  "
                  f"changed={changed}  {flag:9s} [{plan}]")
            rows.append((gp, name, f"{gap:.3f}", str(stops), str(changed), flag, plan))
            if not sane:
                degenerate.append((gp, name))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('gp\talgo\tgap_s\tstops\tchanged\tflag\tplan\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    if args.degenerate_out:
        with open(args.degenerate_out, 'w') as f:
            for gp, name in degenerate:
                f.write(f"{gp}\t{name}\n")
    n_sane = sum(1 for r in rows if r[5] == 'sane')
    print(f"\nsane {n_sane}/{len(rows)}; degenerate: {degenerate}")


if __name__ == '__main__':
    main()
