"""2026-10-18 — Accion del agente en la vuelta 0 (eleccion del compuesto de partida), 400 modelos multi-piloto v7.
Por GP x algoritmo x semilla: 50 resets con rivales estocasticos (compuestos de partida uniformes, como entrenamiento y
validacion) y prediccion determinista; % de accion 0 (no valida: el entorno sortea el compuesto, _handle_lap_0) y
distribucion de las acciones validas. Solo un paso por reset (sin simular la carrera).
    python scripts/gp_drivers_v5hd_20261014.py <este script> <model_dir>   ->  lap0/<model_dir>.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np, pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS, load_gp_config, ALGO_CLASSES, NORM_OBS_KEYS

MD = sys.argv[1]; TREE = os.path.join(ROOT, 'trained_models', 'rl_agents_v7_20261015')
e = [x for x in GP_CONFIGS if x['model_dir'] == MD][0]; gp = e['gp_name']
loader = ModelLoader(); ip = load_gp_config(gp)['initial_positions_list_2024']
rows = []
for algo in ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']:
    for s in range(1, 11):
        mp = os.path.join(TREE, f'seed{s:02d}', MD, algo, 'best_model.zip')
        raw = create_f1_env_all_drivers(gp=gp, driver=e['agent_driver'], deterministic=False, yf_enabled=True,
                                        initial_positions=ip, loader=loader, verbose=False)
        ve = DummyVecEnv([lambda: raw]); vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
        ve = VecNormalize.load(vn, ve) if os.path.exists(vn) else VecNormalize(ve, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
        ve.training = False; ve.norm_reward = False
        m = ALGO_CLASSES[algo].load(mp, env=ve)
        acts = []
        for k in range(50):
            ve.seed(1000 + k); obs = ve.reset()
            kw = dict(state=None, episode_start=np.ones((1,), dtype=bool)) if algo == 'RecurrentPPO' else {}
            a = m.predict(obs, deterministic=True, **kw)[0]
            acts.append(int(np.asarray(a).ravel()[0]))
        acts = np.array(acts)
        rows.append(dict(model_dir=MD, algo=algo, seed=s, pct_a0=100 * (acts == 0).mean(),
                         **{f'pct_a{j}': 100 * (acts == j).mean() for j in range(1, 4)}))
        ve.close()
os.makedirs(os.path.join(HERE, 'lap0'), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'lap0', f'{MD}.csv'), index=False)
print(MD, 'ok')
