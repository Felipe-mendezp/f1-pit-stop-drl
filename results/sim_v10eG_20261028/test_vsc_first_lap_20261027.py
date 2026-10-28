"""2026-10-27 — Prueba del flag F1_VSC_FIRST_LAP_ONLY (v10): con VSC real en las vueltas 10-11, pedir parada en la 11
(2a vuelta de VSC) debe ser no-op con el flag y una parada sin el flag; en la 10 (1a vuelta) para siempre. Con el entorno
del paper (reproduce/env_paper_v3_20261007.sh, sc_reveal). Correr con F1_SIMULATION_DIR=trained_models/simulation_v10_20261027."""
import os, sys
sys.path[:0] = ['.', 'src']
import config, compat; compat.setup_all()
import numpy as np
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from model_loader import ModelLoader
from evaluate_all_gps import load_gp_config

GP, DRV = 'Saudi Arabian Grand Prix', 'Driver_NOR'
loader = ModelLoader()


def run(pit_lap, flag):
    if flag: os.environ['F1_VSC_FIRST_LAP_ONLY'] = '1'
    else: os.environ.pop('F1_VSC_FIRST_LAP_ONLY', None)
    gpc = load_gp_config(GP)
    env = create_f1_env_all_drivers(gp=GP, driver=DRV, deterministic=False, yf_enabled=True,
                                    initial_positions=gpc['initial_positions_list_2024'], loader=loader, verbose=False)
    env.set_evaluation_mode(eval_mode=True, laps_vsc=[10, 11], laps_sc=[], initial_positions=gpc['initial_positions_list_2024'], rival_actions={})
    env.reset(seed=1)
    env.step(2)                       # largada
    pitted = None
    while env.lap_number <= pit_lap:
        lap = env.lap_number
        a = 1 if lap == pit_lap else 0
        before = (env.agent_driver.compound, env.agent_driver.tyrelife)
        env.step(a)
        if lap == pit_lap:
            pitted = env.agent_driver.prev_action != 0
    return pitted


for flag in (False, True):
    print(f'flag={flag}: pide parar en la v.10 (1a VSC) -> paro={run(10, flag)} | en la v.11 (2a VSC) -> paro={run(11, flag)}')
