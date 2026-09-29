import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Recover the agent's starting tyre (lap-0 compound chosen by the deterministic
policy) for each circuit, and save it to results/decision_support/initial_compounds.json.

The starting compound is policy-determined at lap 0, so we just reset the env and
run the first deterministic step (prefix_seed=0, matching the rollouts). Used by
plot_decision_evolution.py to draw the lap-0 tyre square.
"""

import json
import numpy as np
import torch

from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import load_gp_config, build_rival_strategies, GP_CONFIGS, ALGO_CLASSES, NORM_OBS_KEYS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from decision_support_rollouts import batch_obs, action_labels

CIRCUITS = [
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO'),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR'),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO'),
    ('Emilia Romagna Grand Prix', 'Driver_TSU', 'Emilia_Romagna_TSU'),
    ('Hungarian Grand Prix', 'Driver_SAI', 'Hungarian_SAI'),
    ('Belgian Grand Prix', 'Driver_NOR', 'Belgian_NOR'),
    ('Dutch Grand Prix', 'Driver_RUS', 'Dutch_RUS'),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL'),
    ('United States Grand Prix', 'Driver_PIA', 'United_States_PIA'),
]


def initial_compound(gp, driver, model_dir, prefix_seed=0):
    entry = next(e for e in GP_CONFIGS if e['gp_name'] == gp and e['agent_driver'] == driver)
    gp_config = load_gp_config(gp)
    _, initial_compounds = build_rival_strategies(entry)
    loader = ModelLoader()
    raw_env = create_f1_env_all_drivers(gp=gp, driver=driver, deterministic=False,
                                        yf_enabled=True,
                                        initial_positions=gp_config['initial_positions_list_2024'],
                                        loader=loader, verbose=False)
    vec_env = DummyVecEnv([lambda: raw_env])
    mp = os.path.join(str(config.RL_AGENTS_DIR), model_dir, 'RecurrentPPO', 'best_model.zip')
    vn = os.path.join(os.path.dirname(mp), 'vecnormalize_stats.pkl')
    vec_env = VecNormalize.load(vn, vec_env) if os.path.exists(vn) else \
        VecNormalize(vec_env, norm_reward=False, norm_obs_keys=NORM_OBS_KEYS, clip_reward=10000.0)
    vec_env.training = False; vec_env.norm_reward = False
    model = ALGO_CLASSES['RecurrentPPO'].load(mp, env=vec_env)

    np.random.seed(prefix_seed); torch.manual_seed(prefix_seed)
    raw_env.np_random = np.random.default_rng(prefix_seed)
    raw_env.action_space.seed(prefix_seed)
    obs, _ = raw_env.reset(seed=prefix_seed)
    for rival in raw_env.rival_drivers:
        c = initial_compounds.get(rival.name)
        if c is not None:
            rival.compound = c; rival.initial_compound = c
    # lap 0: deterministic compound choice
    norm = vec_env.normalize_obs(batch_obs(obs))
    action, _ = model.predict(norm, state=None, episode_start=np.ones((1,), bool),
                              deterministic=True)
    raw_env.step(int(np.asarray(action).squeeze()))

    comp = int(raw_env.agent_driver.initial_compound)
    labels = action_labels(raw_env.available_compounds)
    avail = raw_env.available_compounds
    act_idx = avail.index(comp) + 1  # 1..n (Hard..Soft)
    return {'compound': comp, 'action': act_idx, 'label': labels[act_idx]}


def main():
    out = {}
    for gp, drv, md in CIRCUITS:
        try:
            info = initial_compound(gp, drv, md)
            out[md] = info
            print(f"{md:22s} start tyre: {info['label']} (C{info['compound']}, action {info['action']})")
        except Exception as e:
            print(f"{md:22s} FAILED: {e}")
    path = os.path.join(str(config.RESULTS_DIR) if hasattr(config, 'RESULTS_DIR') else
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'),
                        'decision_support', 'initial_compounds.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\nWrote {path}")


if __name__ == '__main__':
    main()
