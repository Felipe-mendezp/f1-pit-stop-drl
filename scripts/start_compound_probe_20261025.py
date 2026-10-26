import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-25 — Diagnostico de largada forzada (propuesta de post-entrenamiento, docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md).

Por GP x algoritmo, N carreras completas con la politica determinista (argmax), grilla real 2024, rivales y SC/VSC
simulados, SIN mascara de legalidad. Modos: 'propia' (el agente elige la largada) y 'C<k>' (largada forzada con cada
compuesto disponible; despues decide la politica). Misma semilla por carrera en todos los modos (pareado).
Reporta % de largadas por compuesto en 'propia', y por modo: % descalificadas (un solo compuesto), posicion media
(DSQ = P20, como evaluate_all_gps) y paradas medias.
Correr con el envoltorio v9r2 (igual que la seleccion de semillas):
    python scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py \
        scripts/start_compound_probe_20261025.py --agents <arbol>/selected --out <csv> [--n 200] [--gps ...]
"""
import argparse, collections
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from model_loader import ModelLoader
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from evaluate_all_gps import GP_CONFIGS, load_gp_config

ALGOS = {'PPO': PPO, 'RecurrentPPO': RecurrentPPO}


def race(raw, venv, model, seed, force):
    obs, _ = raw.reset(seed=seed)
    st, es, stops, own = None, np.ones(1, bool), 0, None
    while True:
        o = venv.normalize_obs({k: np.asarray(v)[None] for k, v in obs.items()})
        a, st = model.predict(o, state=st, episode_start=es, deterministic=True)
        es, a = np.zeros(1, bool), int(np.squeeze(a))
        if raw.lap_number == 0:
            own = a
            if force is not None:
                a = raw.available_compounds.index(force) + 1
        elif a:
            stops += 1
        obs, _, done, _, info = raw.step(a)
        if done:
            break
    dsq = not info.get('has_changed_compound', False)
    pos = 20 if dsq else int(info.get('final_position', raw.agent_driver.position))
    return dict(start=raw.agent_driver.initial_compound, own_action=own, dsq=dsq, pos=pos, stops=stops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agents', required=True, help='carpeta con <GP_DRV>/<ALGO>/best_model.zip')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--gps', nargs='*', default=None, help='model_dir de GP_CONFIGS (default: los presentes en --agents)')
    ap.add_argument('--algos', nargs='+', default=list(ALGOS))
    args = ap.parse_args()
    loader, rows = ModelLoader(), []
    for e in GP_CONFIGS:
        d = e['model_dir']
        if (args.gps and d not in args.gps) or not os.path.isdir(os.path.join(args.agents, d)):
            continue
        gp, drv = e['gp_name'], e['agent_driver']
        ip = load_gp_config(gp)['initial_positions_list_2024']
        for algo in args.algos:
            mdir = os.path.join(args.agents, d, algo)
            if not os.path.exists(os.path.join(mdir, 'best_model.zip')):
                continue
            raw = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True, loader=loader,
                                            verbose=False, fast_rivals=True, initial_positions=ip)
            venv = VecNormalize.load(os.path.join(mdir, 'vecnormalize_stats.pkl'), DummyVecEnv([lambda: raw]))
            venv.training, venv.norm_reward = False, False
            model = ALGOS[algo].load(os.path.join(mdir, 'best_model.zip'), env=venv)
            comps = list(raw.available_compounds)
            for force in [None] + comps:
                res = [race(raw, venv, model, 50_000 + s, force) for s in range(args.n)]
                starts = collections.Counter(r['start'] for r in res)
                row = dict(gp=d, algo=algo, modo='propia' if force is None else f'C{force}', n=args.n,
                           dsq_pct=100 * np.mean([r['dsq'] for r in res]), pos=np.mean([r['pos'] for r in res]),
                           paradas=np.mean([r['stops'] for r in res]))
                if force is None:
                    row['largadas'] = ' '.join(f'C{c}:{100 * starts[c] / args.n:.0f}%' for c in comps if starts[c])
                rows.append(row)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv(args.out, index=False)
    print('->', args.out)


if __name__ == '__main__':
    main()
