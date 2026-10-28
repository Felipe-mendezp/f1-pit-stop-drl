"""2026-10-10 — Seleccion de semillas del reentrenamiento v5 (= v4 opcion3 + coeficientes de piloto
corregidos por la Ec. 2; simulador trained_models/simulation_v5_20261010). Copia de
scripts/select_seeds_v4_20261007.py con la misma logica, set de validacion y filtro; solo cambian las
rutas (--calib v5). 2026-10-12: opcion --algos (por defecto PPO RecurrentPPO, como antes) para DQN/A2C/TRPO;
con --algos distinto del default el resumen se escribe en seed_summary_<algos>.csv. 2026-10-12: --calib v5h
(simulation_v5h8_20261012, agentes rl_agents_v5h_20261011, salida results/seed_selection_v5_20261010/v5h/).
2026-10-14: --calib v6 (simulation_v6_20261014, rl_agents_v6_20261014; via scripts/rival_dnf_v6_20261014.py y con
F1_SIMULATION_DIR exportado; ver reproduce/pipeline_v6_20261014.sh). 2026-10-17: opcion --ok-neutral (usada en v7): una
primera parada en las vueltas 1-4 tambien es valida si ocurre con un SC/VSC ya iniciado (en Saudi, con P(SC)=0.70, el
filtro original dejaba a todas las semillas de PPO/RPPO bajo 95 % por paradas bajo SC temprano). Los SC/VSC de cada
carrera se regeneran con la misma semilla de la etapa eval; el resumen va a seed_summary[...]_okneutral.csv. Sin la
opcion, todo igual que antes. 2026-10-17 (tarde): opciones --first-after S, --late X y --min-gap Y (usadas en v7 con
5/7/8, decision del autor: las mismas reglas que el simulador impone a los rivales en f1_env_all_drivers.py). Una
carrera es valida si tiene 1-4 paradas, la primera despues de la vuelta S (o bajo SC/VSC ya iniciado, como
--ok-neutral, que queda implicita), ninguna parada en verde con <= X vueltas restantes (n_laps - vuelta) y ninguna
parada en verde a menos de Y vueltas de la anterior; "en verde" = la vuelta de la parada no esta en las listas SC/VSC
de la carrera. Con X = Y = 0 y S = 4 es el filtro --ok-neutral. El resumen va a seed_summary[...]_r<S><X><Y>.csv.
2026-10-19: --calib v8 (simulador v7, agentes rl_agents_v8_20261018; correr via scripts/rival_start_v8_20261018.py, que
fija los compuestos de partida de rivales Dirichlet y la accion 0 de la vuelta 0 = medio, igual que el entrenamiento).
Ademas los CSV de validacion nuevos tienen la columna a0 (accion del agente en la vuelta 0) y el resumen la
columna a0_cero (fraccion de carreras con accion 0 en la vuelta 0) si el CSV la trae; los CSV antiguos no cambian.
2026-10-19 (noche): opcion --max-dsq D (usada en v8 con 0.01, decision del autor): una semilla pasa solo si ademas su
fraccion de carreras de validacion descalificadas (sin cambio de compuesto) es <= D. Si ninguna pasa, el respaldo es el
de siempre (mayor cumplimiento). El resumen va a seed_summary[...]_dsq<100*D>.csv. Sin la opcion, todo igual.
2026-10-23: --calib v9r2 (simulador v7, agentes rl_agents_v9r2_20261022 = v8 + juegos reutilizables; correr via
scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py, igual que el
entrenamiento; ver reproduce/pipeline_v9r2_20261023.sh). Para pruebas, F1_SELECT_TREE y F1_SELECT_OUT reemplazan el
arbol de agentes y la carpeta de salida (sin ellas, todo igual).
Texto original:

2026-10-07 — [v4 = v3 con el modelo YF v4: simuladores simulation_yfv4_<calib>_20261007, ver scripts/fit_yf_model_v4_20261007.py] Validación y selección de semillas del reentrenamiento v3.

Para cada calibración, GP y algoritmo (PPO, RecurrentPPO) evalúa los modelos de las 10 semillas
(trained_models/rl_agents_v4_<calib>_20261007/seedXX/<GP_piloto>/<algo>/best_model.zip) en un set de
VALIDACIÓN estilo Fig 6 (SC/VSC pre-generados, rivales estocásticos) con una semilla base DISTINTA a la
de la Fig 6 del paper (42). La Fig 7 (carreras reales 2024) no se usa para seleccionar.

Por carrera registra la posición final y las paradas REALES ejecutadas (no cambios de compuesto).
Una carrera "cumple" el filtro si tiene 1-4 paradas y la primera después de la vuelta 4.
Una semilla pasa si cumple en >= --min-ok de las carreras (por defecto 0.95, elegido por Charles 2026-10-07). Entre las que pasan se
elige la de menor posición media; si ninguna pasa, se marca y se elige la de mayor cumplimiento.

Física: env_paper_v3_20261007.sh (F1_SC_REVEAL=1) y el simulador de la calibración. El script fija
F1_SIMULATION_DIR según --calib ANTES de importar config.

    # evaluación (reanudable; un CSV por GP x algoritmo)
    python scripts/select_seeds_v4_20261007.py eval --calib control [--gps Bahrain_ALO] [--n 200] [--jobs 4]
    # resumen + copia de los elegidos a trained_models/rl_agents_v4_<calib>_20261007/selected/
    python scripts/select_seeds_v4_20261007.py select --calib control [--min-ok 0.95]
Salida: results/seed_selection_v4_20261007/<calib>/
"""
import argparse, os, sys, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = {'v5': 'trained_models/simulation_v5_20261010',
       'v5h': 'trained_models/simulation_v5h8_20261012',      # opcion 3 jerarquica en los 8 GP
       'v6': 'trained_models/simulation_v6_20261014',         # 2026-10-14: v6 (correr via scripts/rival_dnf_v6_20261014.py)
       'v7': 'trained_models/simulation_v7_20261015',         # 2026-10-15: v7 (igual que v6: via rival_dnf_v6)
       'v8': 'trained_models/simulation_v7_20261015',         # 2026-10-19: v8 (via rival_start_v8; mismo simulador que v7)
       'v9r2': 'trained_models/simulation_v7_20261015',       # 2026-10-23: v9r2 (via tyre_reuse_v9r2 + tyre_sets_v9 + rival_start_v8)
       'v10': 'trained_models/simulation_v10eGQ_20261028'}     # 2026-10-27: v10 (via v10_wrapper + los envoltorios de v9r2); 28-oct: arbol v10eGQ (edad efectiva; F1_TYRE_SETS/F1_TYRE_STINT_AGES de tyre_ages_v10eGQ)
AGENTS = {'v5': 'rl_agents_v5_20261010', 'v5h': 'rl_agents_v5h_20261011', 'v6': 'rl_agents_v6_20261014',
          'v7': 'rl_agents_v7_20261015', 'v8': 'rl_agents_v8_20261018', 'v9r2': 'rl_agents_v9r2_20261022',
          'v10': 'retrained_v10eGQ_20261028'}
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
    ap.add_argument('--algos', nargs='+', default=ALGOS)
    ap.add_argument('--ok-neutral', action='store_true',
                    help='2026-10-17: primera parada en vueltas 1-4 valida si es bajo SC/VSC ya iniciado')
    ap.add_argument('--first-after', type=int, default=4, help='2026-10-17: primera parada despues de esta vuelta')
    ap.add_argument('--late', type=int, default=0, help='2026-10-17: sin paradas en verde con <= X vueltas restantes (0 = no)')
    ap.add_argument('--min-gap', type=int, default=0, help='2026-10-17: sin paradas en verde a < Y vueltas de la anterior (0 = no)')
    ap.add_argument('--max-dsq', type=float, default=None, help='2026-10-19: maxima fraccion de carreras descalificadas en validacion')
    a = ap.parse_args()
    a.rules = (a.first_after, a.late, a.min_gap) != (4, 0, 0)
    a.ok_neutral = a.ok_neutral or a.rules
    return a


A = parse()
os.environ['F1_SIMULATION_DIR'] = os.path.join(ROOT, SIM[A.calib])
if not os.environ.get('F1_SC_REVEAL'):
    sys.exit('Falta la física v3: source reproduce/env_paper_v3_20261007.sh')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config  # noqa: E402
import compat; compat.setup_all()  # noqa: E402
# 2026-10-14: con un envoltorio (rival_dnf_v6 / gp_drivers) config ya esta importado antes de fijar F1_SIMULATION_DIR
# arriba; hay que pasarlo tambien por fuera. Control de que el simulador cargado es el de --calib:
assert os.path.realpath(str(config.SIMULATION_DIR)) == os.path.realpath(os.path.join(ROOT, SIM[A.calib])), \
    f'config.SIMULATION_DIR={config.SIMULATION_DIR} no es {SIM[A.calib]}: exportar F1_SIMULATION_DIR={SIM[A.calib]}'
if A.calib in ('v6', 'v7', 'v8', 'v9r2', 'v10'):
    assert os.environ.get('F1_RIVAL_DNF'), '--calib v6/v7: correr via scripts/rival_dnf_v6_20261014.py (validacion con abandonos)'
if A.calib in ('v8', 'v9r2', 'v10'):
    assert os.environ.get('F1_RIVAL_START_ALPHA') and os.environ.get('F1_LAP0_ACTION0_MEDIUM'), \
        '--calib v8: correr via scripts/rival_start_v8_20261018.py'
if A.calib in ('v9r2', 'v10'):   # 2026-10-23
    assert os.environ.get('F1_TYRE_SETS') and os.environ.get('F1_TYRE_REUSE') == '1', \
        '--calib v9r2: correr via scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py'
if A.calib == 'v10':    # 2026-10-27
    assert os.environ.get('F1_VSC_FIRST_LAP_ONLY') == '1', '--calib v10: correr via scripts/v10_wrapper_20261027.py'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from environment.f1_env_all_drivers import create_f1_env_all_drivers  # noqa: E402
from model_loader import ModelLoader  # noqa: E402
from evaluate_all_gps import GP_CONFIGS, load_gp_config, ALGO_CLASSES, NORM_OBS_KEYS  # noqa: E402
from evaluate_reward_comparison_stochastic import pregenenerate_race_scenarios  # noqa: E402

TREE = os.path.join(ROOT, 'trained_models', AGENTS[A.calib] + A.tag)
OUT = os.path.join(ROOT, 'results', 'seed_selection_v5_20261010', A.calib + A.tag)
TREE = os.environ.get('F1_SELECT_TREE') or TREE   # 2026-10-23: pruebas fuera de trained_models/
OUT = os.environ.get('F1_SELECT_OUT') or OUT


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
    a0 = []   # 2026-10-19: accion del agente en la vuelta 0
    orig0 = raw_env._handle_lap_0
    raw_env._handle_lap_0 = lambda action: (a0.append(int(action)), orig0(action))[1]
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
    del raw_env._handle_lap_0
    changed = info[0].get('has_changed_compound', False)
    pos = info[0].get('final_position', np.nan) if changed else 20
    return pos, changed, pits, (a0[0] if a0 else -1)


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
            pos, ch, pits, a0 = race(model, vec_env, raw_env, algo, VAL_SEED + i, lv, ls, ip)
            rows.append(dict(seed=s, sim=i, final_position=pos, changed=ch, n_pits=len(pits),
                             first_pit=pits[0] if pits else -1, pits=' '.join(map(str, pits)), a0=a0))
        vec_env.close()
        print(f"  {e['model_dir']} {algo} seed{s:02d}: pos media "
              f"{np.mean([r['final_position'] for r in rows if r['seed'] == s]):.2f}", flush=True)
    missing = sorted(set(A.seeds) - {r['seed'] for r in rows})
    if missing:   # 2026-10-23: sin cache incompleto (antes un arbol vacio dejaba CSV vacios que se reusaban)
        print(f"  {e['model_dir']} {algo}: faltan semillas {missing}, no se guarda {path}", flush=True)
        return None
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def do_eval():
    os.makedirs(OUT, exist_ok=True)
    tasks = [(e, a) for e in entries() for a in A.algos]
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
    loader = ModelLoader() if A.ok_neutral else None
    for e in entries():
        for algo in A.algos:
            path = os.path.join(OUT, f"val_{e['model_dir']}_{algo}.csv")
            if not os.path.exists(path):
                print('sin validación:', path); continue
            df = pd.read_csv(path)
            df['ok'] = (df.n_pits.between(1, 4)) & (df.first_pit > A.first_after)
            if A.ok_neutral:   # 2026-10-17: mismos SC/VSC que en eval_one (misma semilla y n)
                gm = loader.load(e['gp_name'])
                scen = pregenenerate_race_scenarios(n_sims=A.n, n_laps=gm.n_laps, race_vsc_prob=gm.vsc_prob,
                                                    race_sc_prob=gm.sc_prob, base_seed=VAL_SEED * 1000)
                assert df.sim.max() < len(scen), f'--n {A.n} no coincide con la validacion de {path}'
                first_neut = {i: min(list(lv) + list(ls)) if len(lv) + len(ls) else 10 ** 6
                              for i, (lv, ls) in enumerate(scen)}
                df['ok'] |= df.n_pits.between(1, 4) & (df.sim.map(first_neut) <= df.first_pit)
                if A.rules:   # 2026-10-17 (tarde): paradas tardias y dobles en verde
                    neut = [set(lv) | set(ls) for lv, ls in scen]
                    pl = df.pits.fillna('').astype(str).apply(lambda t: [int(float(x)) for x in t.split()])
                    bad = [(A.late and any(gm.n_laps - p <= A.late and p not in neut[i] for p in q)) or
                           (A.min_gap and any(b - a < A.min_gap and b not in neut[i] for a, b in zip(q, q[1:])))
                           for i, q in zip(df.sim, pl)]
                    df['ok'] &= ~np.array(bad, dtype=bool)
            g = df.groupby('seed').agg(pos_media=('final_position', 'mean'),
                                       pos_sd=('final_position', 'std'),
                                       dsq=('changed', lambda x: 1 - x.mean()),
                                       share_ok=('ok', 'mean'),
                                       paradas_media=('n_pits', 'mean')).reset_index()
            if 'a0' in df:   # 2026-10-19
                g['a0_cero'] = df.groupby('seed').a0.apply(lambda x: (x == 0).mean()).values
            g['pasa'] = g.share_ok >= A.min_ok
            if A.max_dsq is not None:   # 2026-10-19
                g['pasa'] &= g.dsq <= A.max_dsq + 1e-12
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
                f.write(f'seed {int(best.seed)}  pos_media_val {best.pos_media:.3f}  share_ok {best.share_ok:.3f} {nota}{" ok-neutral" if A.ok_neutral else ""}{f" reglas {A.first_after}/{A.late}/{A.min_gap}" if A.rules else ""}{f" max-dsq {A.max_dsq}" if A.max_dsq is not None else ""}\n')
    s = pd.concat(summ, ignore_index=True)
    suf = ('' if A.algos == ALGOS else '_' + '_'.join(A.algos)) + (f'_r{A.first_after}{A.late}{A.min_gap}' if A.rules else '_okneutral' if A.ok_neutral else '') + (f'_dsq{100 * A.max_dsq:g}' if A.max_dsq is not None else '')
    s.to_csv(os.path.join(OUT, f'seed_summary{suf}.csv'), index=False)
    pd.set_option('display.width', 200)
    print(s.round(3).to_string(index=False))
    print('\nElegidas:')
    print(s[s.elegida][['model_dir', 'algo', 'seed', 'pos_media', 'share_ok', 'nota']].round(3).to_string(index=False))
    print(f'\nModelos elegidos copiados a {os.path.join(TREE, "selected")} '
          f'(usar F1_RL_AGENTS_DIR=<esa carpeta> en los evaluadores)')


if __name__ == '__main__':
    do_eval() if A.mode == 'eval' else do_select()
