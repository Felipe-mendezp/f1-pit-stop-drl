"""2026-10-19 — diag_20261018.py para DQN. DQN redefine predict (epsilon-greedy), asi que el envoltorio de
BaseAlgorithm.predict de diag_20261018.py no lo ve (trace.csv vacio). Aqui DQN.predict llama a BaseAlgorithm.predict
(ya envuelto) cuando deterministic=True, que es lo mismo que hace DQN.predict en ese caso. Sin cambios en la simulacion.
    ... diag_dqn_20261019.py <args de evaluate_world2024_20261010.py>
"""
import os, sys, runpy
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm


def _predict(self, observation, state=None, episode_start=None, deterministic=False):
    assert deterministic, 'diag DQN: se espera deterministic=True'
    return BaseAlgorithm.predict(self, observation, state, episode_start, deterministic)


DQN.predict = _predict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path = os.path.join(ROOT, 'results', 'world2024_v7_20261015', 'bahrain_trpo_dsq', 'diag_20261018.py')
sys.argv[0] = path
runpy.run_path(path, run_name='__main__')
