import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-07 — Diagnostico: paradas bajo bandera amarilla (SC/VSC) del agente vs rivales.

Pregunta: el agente (DRL o QMIP) decide la accion de la vuelta L con la observacion del
final de L-1, y el SC/VSC de la vuelta L se sortea DENTRO de step(L). Ademas, con
sc_bunching, el pit del agente en la vuelta >= 2 de SC se convierte en no-op. Los rivales
deciden despues del sorteo. Este script mide cuanto pasa eso en los escenarios de las
Figs 6 (SC/VSC y rivales estocasticos) y 7 (eventos y rivales reales 2024).

Solo lee modelos; escribe en results/yf_pits_diag_20261007/. Correr con
`source reproduce/env_paper.sh`.

Uso:
    python scripts/diag_yf_pits_20261007.py --gp 'Belgian Grand Prix' --fig 6 --n-sims 100
Salida: <out>/laps_<fig>_<GP>.csv (una fila por vuelta y politica) y events_<fig>_<GP>.csv
(una fila por evento amarillo y politica).
"""

import argparse
import numpy as np
import pandas as pd
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import (GP_CONFIGS, load_gp_config, build_rival_strategies,
                              ALGO_CLASSES, NORM_OBS_KEYS)
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios
from evaluate_qmip_vs_drl import (build_qmip_params, qmip_choose_start_compound,
                                  qmip_choose_action, qmip_gamma)

# Con F1_SC_REVEAL (reproduce/env_paper_v3_20261007.sh) escribe en otra carpeta
# para no pisar el diagnóstico de referencia (física publicada).
OUT = config.ROOT_DIR / 'results' / ('yf_pits_diag_20261007'
                                     + ('_reveal' if os.environ.get('F1_SC_REVEAL') else ''))


def instrument(raw_env, log):
    """Registra, por vuelta simulada, el estado amarillo y las paradas pedidas/ejecutadas."""
    orig_step, orig_upd = raw_env.step, raw_env._update_all_lap_states
    cur = {}

    def step(action):
        cur.clear()
        cur['lap'] = raw_env.lap_number
        cur['req'] = int(np.asarray(action).item()) if raw_env.lap_number >= 1 else 0
        return orig_step(action)

    def upd(agent_action, rival_actions, is_yf):
        alive = sum(1 for d in raw_env.rival_drivers if not d.is_dnf)
        log.append(dict(lap=raw_env.lap_number, sc=int(raw_env.sc > 0), vsc=int(raw_env.vsc > 0),
                        first_yf=int(bool(raw_env._is_first_yf_lap)),
                        sc_idx=int(getattr(raw_env, '_sc_lap_idx', 0)),
                        req_pit=int(cur.get('req', 0) != 0), exe_pit=int(agent_action != 0),
                        riv_pits=int(np.sum(np.asarray(rival_actions) != 0)), riv_alive=alive))
        return orig_upd(agent_action, rival_actions, is_yf)

    raw_env.step, raw_env._update_all_lap_states = step, upd


def load_drl(algo, entry, raw_env):
    vec_env = DummyVecEnv([lambda: raw_env])
    path = os.path.join(str(config.RL_AGENTS_DIR), entry['model_dir'], algo, 'best_model.zip')
    vn = os.path.join(os.path.dirname(path), 'vecnormalize_stats.pkl')
    if os.path.exists(vn):
        vec_env = VecNormalize.load(vn, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS,
                               clip_reward=10000.0)
    vec_env.training = False
    vec_env.norm_reward = False
    return ALGO_CLASSES[algo].load(path, env=vec_env), vec_env


def race_drl(model, vec_env, raw_env, algo, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed); raw_env.action_space.seed(seed)
    obs = vec_env.reset()
    lstm = None; start = np.ones((1,), dtype=bool) if algo == 'RecurrentPPO' else None
    done = False
    while not done:
        if algo == 'RecurrentPPO':
            a, lstm = model.predict(obs, state=lstm, episode_start=start, deterministic=True)
            start = np.zeros((1,), dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, d, info = vec_env.step(a)
        done = d[0]
    return info[0].get('final_position', np.nan)


def race_qmip(raw_env, qp, seed, p=3):
    np.random.seed(seed); raw_env.np_random = np.random.default_rng(seed)
    raw_env.reset()
    a, t0 = qmip_choose_start_compound(qp['betas_dict'], qp['time_stop'], raw_env.n_laps,
                                       qp['qmip_to_compound'], qp['available_compounds'], p=p)
    _, _, term, trunc, info = raw_env.step(a)
    t_prev = [t0]; done = term or trunc
    while not done:
        cur = qp['compound_to_qmip'].get(raw_env.agent_driver.compound, 'Medium')
        gamma, yf_now = qmip_gamma(raw_env)
        a = qmip_choose_action(qp['betas_dict'], qp['time_stop'], raw_env.laps_left, cur,
                               raw_env.agent_driver.tyrelife, t_prev, raw_env.n_laps, gamma,
                               qp['qmip_to_compound'], qp['available_compounds'], p=p,
                               yf_now=yf_now, time_stop_yf=qp.get('time_stop_yf'))
        _, _, term, trunc, info = raw_env.step(a)
        done = term or trunc
        nm = qp['compound_to_qmip'].get(raw_env.agent_driver.compound, 'Medium')
        if nm not in t_prev:
            t_prev.append(nm)
    return info.get('final_position', np.nan)


def events(df):
    """Un evento = racha de vueltas amarillas consecutivas del mismo tipo."""
    rows = []
    for (pol, sim), g in df.groupby(['policy', 'sim']):
        g = g.sort_values('lap').reset_index(drop=True)
        g['typ'] = np.where(g.sc == 1, 'SC', np.where(g.vsc == 1, 'VSC', ''))
        i = 0
        while i < len(g):
            if g.typ[i] == '':
                i += 1; continue
            j = i
            while j + 1 < len(g) and g.typ[j + 1] == g.typ[i] and g.lap[j + 1] == g.lap[j] + 1:
                j += 1
            ev = g.iloc[i:j + 1]
            post = g[(g.lap > g.lap[j]) & (g.lap <= g.lap[j] + 2)]
            rows.append(dict(policy=pol, sim=sim, typ=g.typ[i], start=int(g.lap[i]), length=j - i + 1,
                             ag_exe_first=int(ev.exe_pit.iloc[0]),
                             ag_exe_later=int(ev.exe_pit.iloc[1:].sum() > 0),
                             ag_req_masked=int(((ev.req_pit == 1) & (ev.exe_pit == 0)).sum() > 0),
                             ag_pit_post2=int(post.exe_pit.sum() > 0),
                             riv_share_event=ev.riv_pits.sum() / max(ev.riv_alive.iloc[0], 1),
                             riv_share_first=ev.riv_pits.iloc[0] / max(ev.riv_alive.iloc[0], 1)))
            i = j + 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gp', required=True)
    ap.add_argument('--fig', type=int, choices=[6, 7], required=True)
    ap.add_argument('--n-sims', type=int, default=100)
    ap.add_argument('--policies', nargs='+', default=['PPO', 'RecurrentPPO', 'QMIP'])
    ap.add_argument('--base-seed', type=int, default=42)
    args = ap.parse_args()

    entry = next(e for e in GP_CONFIGS if e['gp_name'] == args.gp)
    gcfg = load_gp_config(args.gp)
    ip = gcfg['initial_positions_list_2024']
    loader = ModelLoader()
    gm = loader.load(args.gp)
    if args.fig == 6:
        # Mismos escenarios que evaluate_qmip_vs_drl.py (seed base*1000, sims seed base+i)
        scen = pregenenerate_race_scenarios(n_sims=args.n_sims, n_laps=gm.n_laps,
                                            race_vsc_prob=gm.vsc_prob, race_sc_prob=gm.sc_prob,
                                            base_seed=args.base_seed * 1000)
        riv, init = {}, None
    else:
        scen = [(gcfg['laps_vsc_2024'], gcfg['laps_sc_2024'])] * args.n_sims
        riv, init = build_rival_strategies(entry)

    log_all, fin = [], []
    for pol in args.policies:
        raw_env = create_f1_env_all_drivers(gp=args.gp, driver=entry['agent_driver'],
                                            deterministic=False, yf_enabled=True,
                                            initial_positions=ip, loader=loader, verbose=False)
        log = []
        instrument(raw_env, log)
        if pol != 'QMIP':
            model, vec_env = load_drl(pol, entry, raw_env)
        else:
            qp = build_qmip_params(loader, args.gp)
        for i, (lv, ls) in enumerate(scen):
            kw = dict(eval_mode=True, laps_vsc=lv, laps_sc=ls, initial_positions=ip, rival_actions=riv)
            if init is not None:
                kw['rival_initial_compounds'] = init
            raw_env.set_evaluation_mode(**kw)
            n0 = len(log)
            pos = (race_qmip(raw_env, qp, args.base_seed + i) if pol == 'QMIP'
                   else race_drl(model, vec_env, raw_env, pol, args.base_seed + i))
            for r in log[n0:]:
                r.update(policy=pol, sim=i)
            fin.append(dict(policy=pol, sim=i, final_position=pos))
        log_all += log
        print(f'{args.gp} fig{args.fig} {pol}: {len(scen)} carreras, '
              f'pos media {np.nanmean([f["final_position"] for f in fin if f["policy"] == pol]):.2f}',
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    tag = f"fig{args.fig}_{args.gp.replace(' Grand Prix', '').replace(' ', '_')}"
    df = pd.DataFrame(log_all).merge(pd.DataFrame(fin), on=['policy', 'sim'])
    df.to_csv(OUT / f'laps_{tag}.csv', index=False)
    ev = events(df)
    ev.to_csv(OUT / f'events_{tag}.csv', index=False)
    if len(ev):
        print(ev.groupby(['policy', 'typ']).mean(numeric_only=True).round(3).to_string())


if __name__ == '__main__':
    main()
