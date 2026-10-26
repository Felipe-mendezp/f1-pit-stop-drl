import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-25 — Post-entrenamiento (fine-tuning) de PPO / RecurrentPPO con largada sorteada
(docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md).

Parte de best_model.zip + vecnormalize_stats.pkl de un agente ya seleccionado (solo lectura) y sigue entrenando con
F1_START_MIX = p: en una fraccion p de los episodios el entorno juega la vuelta 0 con un compuesto uniforme (ver
F1EnvAllDrivers.reset). Mismos entornos que el entrenamiento original (create_env_factory, 10 SubprocVecEnv, grilla
real 2024, misma fisica), misma evaluacion periodica (SyncedEvalCallback + parada temprana + stats de VecNormalize del
mejor modelo), pero el entorno de evaluacion tambien usa la mezcla. Tasa de aprendizaje constante (--lr).
Escribe <out-root>/<GP_DRV>/<ALGO>/ (best_model.zip, vecnormalize_stats.pkl, evaluations.npz, FINETUNE_FROM.txt).
Correr con el envoltorio v9r2, como el entrenamiento original:
    python scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py \
        scripts/finetune_startmix_20261025.py --agents trained_models/rl_agents_v9r2_20261022/selected \
        --gp 'Saudi Arabian Grand Prix' --algo PPO --lr 1e-4 --seed 1 --fast-env --torch-threads 4 \
        --following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5 \
        --out-root trained_models/retrained_v9r2_startmix_20261025/seed01
"""
import argparse

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from train_rl_agents import create_env_factory, create_eval_callback, get_initial_positions, seed_all, TRAINING_CONFIG
from model_loader import ModelLoader
from evaluate_all_gps import GP_CONFIGS

ALGOS = {'PPO': PPO, 'RecurrentPPO': RecurrentPPO}
GP_DRIVER = {e['gp_name']: (e['agent_driver'], e['model_dir']) for e in GP_CONFIGS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agents', required=True, help='carpeta con <GP_DRV>/<ALGO>/best_model.zip (solo lectura)')
    ap.add_argument('--gp', required=True)
    ap.add_argument('--algo', required=True, choices=list(ALGOS))
    ap.add_argument('--lr', type=float, required=True, help='tasa de aprendizaje constante del post-entrenamiento')
    ap.add_argument('--start-mix', type=float, default=0.5, help='prob. de largada sorteada (F1_START_MIX)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--timesteps', type=int, default=1_500_000)
    ap.add_argument('--eval-freq', type=int, default=5_000,
                    help='llamadas por evaluacion (x n_envs pasos); el original usa 15 000 = cada 150 000 pasos')
    ap.add_argument('--n-envs', type=int, default=None)
    ap.add_argument('--torch-threads', type=int, default=None)
    ap.add_argument('--vec', choices=['subproc', 'dummy'], default='subproc',
                    help='dummy = los 10 entornos en el proceso principal (mismo buffer y mismas semillas; menos CPU y RAM)')
    ap.add_argument('--fast-env', action='store_true')
    ap.add_argument('--following-gap-scale', type=float, default=0.0)
    ap.add_argument('--sc-bunching', action='store_true')
    ap.add_argument('--sc-bunch-delta', type=float, default=1.5)
    ap.add_argument('--out-root', required=True)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    assert os.sep + 'rl_agents' not in out_root and os.sep + 'simulation' not in out_root, \
        'no escribir en trained_models/rl_agents* ni trained_models/simulation*'
    # Antes de crear los SubprocVecEnv: los workers heredan la variable y la fabrica del entorno la lee
    os.environ['F1_START_MIX'] = str(args.start_mix)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    driver, model_dir = GP_DRIVER[args.gp]
    src = os.path.join(args.agents, model_dir, args.algo)
    algo_dir = os.path.join(out_root, model_dir, args.algo)
    os.makedirs(algo_dir, exist_ok=True)
    n_envs = args.n_envs or TRAINING_CONFIG['n_envs']

    seed_all(args.seed)
    env_factory = create_env_factory(
        args.gp, driver, ModelLoader(), get_initial_positions(args.gp), reward_type='mix',
        fast_rivals=args.fast_env, following_gap_scale=args.following_gap_scale,
        sc_bunching=args.sc_bunching, sc_bunch_delta=args.sc_bunch_delta)
    vcls = DummyVecEnv if args.vec == 'dummy' else SubprocVecEnv
    vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=vcls, seed=args.seed)
    eval_vec_env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=vcls, seed=args.seed + 10_000)
    # Estadisticas del agente original: siguen actualizandose en el de entrenamiento (como en el original, PPO y
    # RPPO con norm_reward); el de evaluacion queda fijo y se sincroniza antes de cada evaluacion.
    stats = os.path.join(src, 'vecnormalize_stats.pkl')
    vec_env = VecNormalize.load(stats, vec_env)
    vec_env.training, vec_env.norm_reward = True, True
    eval_vec_env = VecNormalize.load(stats, eval_vec_env)
    eval_vec_env.training, eval_vec_env.norm_reward = False, False

    lr = args.lr
    model = ALGOS[args.algo].load(
        os.path.join(src, 'best_model.zip'), env=vec_env, device='cpu',
        custom_objects={'learning_rate': lr, 'lr_schedule': lambda _: lr},
        tensorboard_log=os.path.join(out_root, model_dir, 'tensorboard'), verbose=1)
    model.set_random_seed(args.seed)
    assert abs(model.lr_schedule(1.0) - lr) < 1e-12 and abs(model.lr_schedule(0.0) - lr) < 1e-12

    with open(os.path.join(algo_dir, 'FINETUNE_FROM.txt'), 'w') as f:
        f.write(f'desde {os.path.abspath(src)}\n'
                + (open(os.path.join(src, 'SELECTED_SEED.txt')).read().strip() + '\n'
                   if os.path.exists(os.path.join(src, 'SELECTED_SEED.txt')) else '')
                + f'start_mix {args.start_mix}  lr {lr}  seed {args.seed}  timesteps {args.timesteps}  '
                  f'eval_freq {args.eval_freq}x{n_envs}\n')
    print(f'FINETUNE {args.algo} | {args.gp} | seed {args.seed} | start_mix {args.start_mix} | lr {lr} '
          f'| vec {args.vec} | n_steps {model.n_steps} batch {model.batch_size} ent_coef {model.ent_coef} -> {algo_dir}', flush=True)

    eval_callback = create_eval_callback(
        eval_vec_env, save_path=algo_dir, log_path=algo_dir, eval_freq=args.eval_freq,
        n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
        early_stop_patience=TRAINING_CONFIG['early_stop_patience'])
    model.learn(args.timesteps, callback=eval_callback, reset_num_timesteps=True, tb_log_name=args.algo)

    vec_env.save(os.path.join(algo_dir, 'vecnormalize_stats_final.pkl'))
    if not os.path.exists(os.path.join(algo_dir, 'vecnormalize_stats.pkl')):
        vec_env.save(os.path.join(algo_dir, 'vecnormalize_stats.pkl'))
    vec_env.close()
    eval_vec_env.close()
    ev = np.load(os.path.join(algo_dir, 'evaluations.npz'))
    r = ev['results'].mean(1)
    print(f'OK {args.algo} {args.gp}: {len(r)} evaluaciones, mejor {r.max():.1f} en {ev["timesteps"][r.argmax()]} pasos',
          flush=True)


if __name__ == '__main__':
    main()
