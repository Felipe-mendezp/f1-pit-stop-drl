import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
2026-10-07 — Test del modelo YF v4 dentro del entorno multi-piloto.

Corre carreras estocásticas (modo entrenamiento, política aleatoria) sobre
trained_models/simulation_yfv4_<calib>_20261007 y comprueba que cada tiempo de vuelta
bajo SC/VSC de cada auto es exactamente el que dan los coeficientes de
data/yf_model_v4_20261007.csv: SC (o VSC) + PitIn*PitIn_v4 + PitOut*PitOut_v4
(con --sc-bunching la 1a vuelta de SC usa el SC plano). No escribe nada.

    source reproduce/env_paper_v3_20261007.sh
    python scripts/test_yfv4_env_20261007.py --calib opcion2 --n 20
"""

import argparse

p = argparse.ArgumentParser()
p.add_argument('--calib', default='control', choices=['control', 'opcion2', 'opcion3'])
p.add_argument('--n', type=int, default=20, help='carreras por GP')
A = p.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['F1_SIMULATION_DIR'] = os.path.join(ROOT, 'trained_models', f'simulation_yfv4_{A.calib}_20261007')

import config  # noqa: E402
import compat; compat.setup_all()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from environment.f1_env_all_drivers import create_f1_env_all_drivers  # noqa: E402
from model_loader import ModelLoader  # noqa: E402
from evaluate_all_gps import GP_CONFIGS  # noqa: E402

COEF = pd.read_csv(config.DATA_DIR / 'yf_model_v4_20261007.csv').set_index('GP')


def main():
    assert os.environ.get('F1_SC_REVEAL'), 'source reproduce/env_paper_v3_20261007.sh'
    loader = ModelLoader(str(config.SIMULATION_DIR))
    rng = np.random.default_rng(0)
    tot = {'SC': 0, 'VSC': 0, 'pit': 0, 'bad': 0}
    for e in GP_CONFIGS:
        gp = e['gp_name']
        if gp not in COEF.index or 'Emilia' in gp:
            continue
        c = COEF.loc[gp]
        env = create_f1_env_all_drivers(gp=gp, driver=e['agent_driver'], deterministic=False, loader=loader)
        env.sc_bunching = True
        orig = env._simulate_lap_times
        n = {'SC': 0, 'VSC': 0, 'pit': 0, 'bad': 0}

        def wrapped(is_yf):
            lts = orig(is_yf)
            if is_yf:
                st = 'SC' if env.sc > 0 else 'VSC'
                for d, lt in zip(env._ordered_drivers, lts):
                    if d.is_dnf:
                        continue
                    ls = d.lap_state
                    pin, pout = float(ls.get('PitIn', 0)), float(ls.get('PitOut', 0))
                    want = c[st] + pin * c.PitIn + pout * c.PitOut
                    if st == 'VSC' and env._is_first_yf_lap:
                        want += c.VSC_firstlap
                    n[st] += 1
                    n['pit'] += int(pin + pout > 0)
                    if abs(lt - want) > 1e-2:
                        n['bad'] += 1
            return lts

        env._simulate_lap_times = wrapped
        for k in range(A.n):
            env.reset(seed=int(rng.integers(1 << 30)))
            done = False
            while not done:
                a = int(rng.integers(0, 4)) if rng.random() < 0.08 else 0
                _, _, term, trunc, _ = env.step(a)
                done = term or trunc
        print(f'  {gp:26s} vueltas-auto SC {n["SC"]:5d}  VSC {n["VSC"]:4d}  con pit {n["pit"]:3d}  '
              f'fuera de tolerancia {n["bad"]}')
        for kk in tot:
            tot[kk] += n[kk]
    print(f'TOTAL {tot}')
    assert tot['bad'] == 0 and tot['SC'] > 0 and tot['VSC'] > 0 and tot['pit'] > 0
    print('OK')


if __name__ == '__main__':
    main()
