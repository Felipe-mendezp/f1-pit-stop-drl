"""2026-10-07 — Validación y selección de semillas del reentrenamiento v3.

Para cada calibración, GP y algoritmo (PPO, RecurrentPPO) evalúa los modelos de las 10 semillas
(trained_models/rl_agents_v3_<calib>_20261007/seedXX/<GP_piloto>/<algo>/best_model.zip) en un set de
VALIDACIÓN estilo Fig 6 (SC/VSC pre-generados, rivales estocásticos) con una semilla base DISTINTA a la
de la Fig 6 del paper (42). La Fig 7 (carreras reales 2024) no se usa para seleccionar.

Por carrera registra la posición final y las paradas REALES ejecutadas (no cambios de compuesto).
Una carrera "cumple" el filtro si tiene 1-4 paradas y la primera después de la vuelta 4.
Una semilla pasa si cumple en >= --min-ok de las carreras (por defecto 0.95, elegido por Charles 2026-10-07). Entre las que pasan se
elige la de menor posición media; si ninguna pasa, se marca y se elige la de mayor cumplimiento.

Física: env_paper_v3_20261007.sh (F1_SC_REVEAL=1) y el simulador de la calibración. El script fija
F1_SIMULATION_DIR según --calib ANTES de importar config.

    # evaluación (reanudable; un CSV por GP x algoritmo)
    python scripts/select_seeds_v3_20261007.py eval --calib control [--gps Bahrain_ALO] [--n 200] [--jobs 4]
    # resumen + copia de los elegidos a trained_models/rl_agents_v3_<calib>_20261007/selected/
    python scripts/select_seeds_v3_20261007.py select --calib control [--min-ok 0.95]
Salida: results/seed_selection_v3_20261007/<calib>/
"""
import argparse, os, sys, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = {'control': 'trained_models/simulation',
       'opcion2': 'trained_models/simulation_tyre_opcion2_20261004',
       'opcion3': 'trained_models/simulation_tyre_opcion3_20261004'}
ALGOS = ['PPO', 'RecurrentPPO']
VAL_SEED = 2026          # Fig 6 del paper usa 42


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'select'])
    ap.add_argument('--calib', required=True, choices=list(SIM))
    ap.add_argument('--gps', nargs='+', default=None, help='model_dir, p.ej. Bahrain_ALO')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--n', type=int, default=200, help='carreras de validación por modelo')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--min-ok', type=float, default=0.95)
    ap.add_argument('--tag', default='')
    return ap.parse_args()


A = parse()
os.environ['F1_SIMULATION_DIR'] = os.path.join(ROOT, SIM[A.calib])
if not os.environ.get('F1_SC_REVEAL'):
    sys.exit('Falta la física v3: source reproduce/env_paper_v3_20261007.sh')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config  # noqa: E402
import compat; compat.setup_all()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from environment.f1_env_all_drivers import create_f1_env_all_drivers  # noqa: E402
from model_loader import ModelLoader  # noqa: E402
from evaluate_all_gps import GP_CONFIGS, load_gp_config, ALGO_CLASSES, NORM_OBS_KEYS  # noqa: E402
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios  # noqa: E402

TREE = os.path.join(ROOT, 'trained_models', f'rl_agents_v3_{A.calib}_20261007{A.tag}')
OUT = os.path.join(ROOT, 'results', 'seed_selection_v3_20261007', A.calib + A.tag)


def entries():
    es = [e for e in GP_CONFIGS if 'Emilia' not in e['gp_name']]
    return [e for e in es if A.gps is None or e['model_dir'] in A.gps]


def race(model, vec_env, raw_env, algo, seed, lv, ls, ip):
    raw_env.set_evaluation_mode(eval_mode=True, laps_vsc=lv, laps_sc=ls, initial_positions=ip,
                                rival_actions={})
    np.random.seed(seed); torch.manual_seed(seed)
    raw_env.np_random = np.random.default_rng(seed); raw_env.action_space.seed(seed)
    pits = []
    orig = raw_env._update_all_lap_states

    def upd(agent_action, rival_actions, is_yf):
        if agent_action != 0:
            pits.append(raw_env.lap_number)
        return orig(agent_action, rival_actions, is_yf)

    raw_env._update_all_lap_states = upd
    obs = vec_env.reset()
    lstm, start = None, (np.ones((1,), dtype=bool) if algo == 'RecurrentPPO' else None)
    done, info = False, [{}]
    while not done:
        if algo == 'RecurrentPPO':
            a, lstm = model.predict(obs, state=lstm, episode_start=start, deterministic=True)
            start = np.zeros((1,), dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, d, info = vec_env.step(a)
        done = d[0]
    raw_env._update_all_lap_states = orig
    changed = info[0].get('has_changed_compound', False)
    pos = info[0].get('final_position', np.nan) if changed else 20
    return pos, changed, pits


def eval_one(e, algo, loader):
    path = os.path.join(OUT, f"val_{e['model_dir']}_{algo}.csv")
    if os.path.exists(path):
        return path
    gp = e['gp_name']
    ip = load_gp_config(gp)['initial_positions_list_2024']
    gm = loader.load(gp)
    scen = pregenenerate_race_scenarios(n_sims=A.n, n_laps=gm.n_laps, race_vsc_prob=gm.vsc_prob,
                                        race_sc_prob=gm.sc_prob, base_seed=VAL_SEED * 1000)
    rows = []
    for s in A.seeds:
        mp = os.path.join(TREE, f'seed{s:02d}', e['model_dir'], algo, 'best_model.zip')
        if not os.path.exists(mp):
            print(f'  falta {mp}', flush=True); continue
        raw_env = create_f1_env_all_drivers(gp=gp, driver=e['agent_driver'], deterministic=False,
                                            yf_enabled=True, initial_positions=ip, loader=loader,
                                            verbose=False)
        vec_env = DummyVecEnv([lambda: raw_env])
        vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
        vec_env = (VecNormalize.load(vn, vec_env) if os.path.exists(vn) else
                   VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0))
        vec_env.training = False; vec_env.norm_reward = False
        model = ALGO_CLASSES[algo].load(mp, env=vec_env)
        for i, (lv, ls) in enumerate(scen):
            pos, ch, pits = race(model, vec_env, raw_env, algo, VAL_SEED + i, lv, ls, ip)
            rows.append(dict(seed=s, sim=i, final_position=pos, changed=ch, n_pits=len(pits),
                             first_pit=pits[0] if pits else -1, pits=' '.join(map(str, pits))))
        vec_env.close()
        print(f"  {e['model_dir']} {algo} seed{s:02d}: pos media "
              f"{np.mean([r['final_position'] for r in rows if r['seed'] == s]):.2f}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def do_eval():
    os.makedirs(OUT, exist_ok=True)
    tasks = [(e, a) for e in entries() for a in ALGOS]
    if A.jobs <= 1:
        loader = ModelLoader()
        for e, a in tasks:
            eval_one(e, a, loader)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(A.jobs) as ex:
            list(ex.map(_worker, tasks))


def _worker(task):
    return eval_one(task[0], task[1], ModelLoader())


def do_select():
    summ = []
    for e in entries():
        for algo in ALGOS:
            path = os.path.join(OUT, f"val_{e['model_dir']}_{algo}.csv")
            if not os.path.exists(path):
                print('sin validación:', path); continue
            df = pd.read_csv(path)
            df['ok'] = (df.n_pits.between(1, 4)) & (df.first_pit > 4)
            g = df.groupby('seed').agg(pos_media=('final_position', 'mean'),
                                       pos_sd=('final_position', 'std'),
                                       dsq=('changed', lambda x: 1 - x.mean()),
                                       share_ok=('ok', 'mean'),
                                       paradas_media=('n_pits', 'mean')).reset_index()
            g['pasa'] = g.share_ok >= A.min_ok
            cand = g[g.pasa]
            if len(cand):
                best, nota = cand.sort_values('pos_media').iloc[0], ''
            else:
                best, nota = g.sort_values(['share_ok', 'pos_media'], ascending=[False, True]).iloc[0], 'NINGUNA PASA'
            g['elegida'] = g.seed == best.seed
            g.insert(0, 'algo', algo); g.insert(0, 'model_dir', e['model_dir'])
            g['nota'] = np.where(g.elegida, nota, '')
            summ.append(g)
            src = os.path.join(TREE, f'seed{int(best.seed):02d}', e['model_dir'], algo)
            dst = os.path.join(TREE, 'selected', e['model_dir'], algo)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('ck'))
            with open(os.path.join(dst, 'SELECTED_SEED.txt'), 'w') as f:
                f.write(f'seed {int(best.seed)}  pos_media_val {best.pos_media:.3f}  share_ok {best.share_ok:.3f} {nota}\n')
    s = pd.concat(summ, ignore_index=True)
    s.to_csv(os.path.join(OUT, 'seed_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(s.round(3).to_string(index=False))
    print('\nElegidas:')
    print(s[s.elegida][['model_dir', 'algo', 'seed', 'pos_media', 'share_ok', 'nota']].round(3).to_string(index=False))
    print(f'\nModelos elegidos copiados a {os.path.join(TREE, "selected")} '
          f'(usar F1_RL_AGENTS_DIR=<esa carpeta> en los evaluadores)')


if __name__ == '__main__':
    do_eval() if A.mode == 'eval' else do_select()
