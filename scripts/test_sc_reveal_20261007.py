import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-07 — Test del flag `sc_reveal` (F1_SC_REVEAL) del entorno multi-piloto.

Con una política FIJA (acciones que no dependen de la observación), activar el
flag solo debe cambiar CUÁNDO se ve el SC/VSC, no la carrera: mismos eventos,
mismas paradas ejecutadas, mismos tiempos y posición final. Se prueba en
  (a) modo estocástico (el de entrenamiento: sorteo interno del SC/VSC), y
  (b) modo eval (Fig 6: escenarios pre-generados; Fig 7: eventos 2024).
Además verifica el contador observado con el flag: SC 4,3,2,1 / VSC 2,1, con
4 (2) solo en la 1a vuelta del evento.

No escribe nada. Uso (con source reproduce/env_paper.sh):
    python scripts/test_sc_reveal_20261007.py --n 200
"""

import argparse
import numpy as np

from environment.f1_env_all_drivers import create_f1_env_all_drivers, SC_DURATION, VSC_DURATION
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS, load_gp_config, build_rival_strategies
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios


def policy(lap, n_laps, k):
    """Acciones fijas: compuesto de salida y 2 paradas en vueltas que varían con k."""
    if lap == 0:
        return 1 + k % 3
    a, b = 3 + (k * 7) % (n_laps // 2), n_laps // 2 + (k * 5) % (n_laps // 3)
    if lap == a:
        return 1 + (k + 1) % 3
    if lap == b:
        return 1 + (k + 2) % 3
    return 0


def run(env, k, seed, eval_kw):
    log, obs_log = [], []
    orig = env._update_all_lap_states

    def upd(agent_action, rival_actions, is_yf):
        log.append((env.lap_number, int(env.sc > 0), int(env.vsc > 0),
                    int(bool(env._is_first_yf_lap)), int(agent_action),
                    tuple(int(x) for x in np.asarray(rival_actions))))
        return orig(agent_action, rival_actions, is_yf)

    env._update_all_lap_states = upd
    # El camino legacy (sin flag) no limpia _sc_lap_idx en reset(): si la
    # carrera anterior terminó bajo SC y esta tiene SC en la vuelta 1, la trata
    # como 2a vuelta de SC (pit anulado, compactación). Afecta 4/4500
    # escenarios de la Fig 6 publicada y se deja así para reproducirla; con
    # sc_reveal no ocurre (se reinicia en la vuelta 0). Se limpia aquí para
    # comparar ambos caminos sobre la misma carrera.
    if not getattr(env, 'sc_reveal', False):
        env._sc_lap_idx = 0
    if eval_kw is not None:
        env.set_evaluation_mode(**eval_kw)
        env.np_random = np.random.default_rng(seed)
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs_log.append((env.lap_number, int(obs['sc']), int(obs['vsc'])))
        obs, r, term, trunc, info = env.step(policy(env.lap_number, env.n_laps, k))
        done = term or trunc
    env._update_all_lap_states = orig
    return log, obs_log, info['final_position'], round(info['total_time'], 6)


def check_counter(log, obs_log):
    """Con reveal: la obs antes de decidir la vuelta L describe la vuelta L."""
    phys = {lap: (sc, vsc, first) for lap, sc, vsc, first, *_ in log}
    bad = 0
    for lap, osc, ovsc in obs_log:
        if lap == 0:
            continue
        sc, vsc, first = phys[lap]
        ok = ((osc > 0) == bool(sc) and (ovsc > 0) == bool(vsc)
              and (osc == SC_DURATION) == bool(sc and first)
              and (ovsc == VSC_DURATION) == bool(vsc and first))
        bad += not ok
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--gps', nargs='+', default=None)
    args = ap.parse_args()
    loader = ModelLoader()
    entries = [e for e in GP_CONFIGS if args.gps is None or e['gp_name'] in args.gps]
    total_bad = 0
    for e in entries:
        gp = e['gp_name']
        gcfg = load_gp_config(gp)
        ip = gcfg['initial_positions_list_2024']
        gm = loader.load(gp)
        scen6 = pregenenerate_race_scenarios(n_sims=args.n, n_laps=gm.n_laps,
                                             race_vsc_prob=gm.vsc_prob, race_sc_prob=gm.sc_prob,
                                             base_seed=42 * 1000)
        riv, init = build_rival_strategies(e)
        modes = {
            'estocastico': lambda i: None,
            'fig6': lambda i: dict(eval_mode=True, laps_vsc=scen6[i][0], laps_sc=scen6[i][1],
                                   initial_positions=ip, rival_actions={}),
            'fig7': lambda i: dict(eval_mode=True, laps_vsc=gcfg['laps_vsc_2024'],
                                   laps_sc=gcfg['laps_sc_2024'], initial_positions=ip,
                                   rival_actions=riv, rival_initial_compounds=init),
        }
        envs = {}
        for flag in (False, True):
            env = create_f1_env_all_drivers(gp=gp, driver=e['agent_driver'], deterministic=False,
                                            yf_enabled=True, initial_positions=ip, loader=loader,
                                            verbose=False)
            env.sc_reveal = flag
            envs[flag] = env
        for mname, kw in modes.items():
            n = args.n if mname != 'fig7' else min(args.n, 50)
            diff = cbad = n_sc = n_vsc = 0
            for i in range(n):
                seed = 42 + i
                a = run(envs[False], i, seed, kw(i))
                b = run(envs[True], i, seed, kw(i))
                if (a[0], a[2], a[3]) != (b[0], b[2], b[3]):
                    diff += 1
                cbad += check_counter(b[0], b[1])
                n_sc += any(x[1] for x in b[0]); n_vsc += any(x[2] for x in b[0])
            total_bad += diff + cbad
            print(f'{gp:28s} {mname:12s} n={n:4d} carreras_con_SC={n_sc:4d} con_VSC={n_vsc:4d} '
                  f'| distintas={diff} contador_mal={cbad}', flush=True)
    print('OK' if total_bad == 0 else f'FALLA: {total_bad}')


if __name__ == '__main__':
    main()
