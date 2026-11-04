"""2026-11-03 — Tiempo de una resolucion de la QMIP y de una decision de la politica PPO, en un proceso. La QMIP se resuelve con el
inventario de juegos de cada GP al inicio (eligiendo compuesto de partida), con 40 vueltas por correr y al final de la carrera.
    (entorno de ../sc_second_lap_check_v10g_20261103/run_20261103.sh) python results/runtimes_20261103/solve_times_20261103.py"""
import sys, os, time; R = os.getcwd(); sys.path[:0] = [R, f'{R}/src', f'{R}/scripts']
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from model_loader import ModelLoader
from optimization.qmip import solve_MIP_GUROBI, solve_MIP_GUROBI_start
from optimization.utils_qmip import extract_simplified_calculator
from stable_baselines3 import PPO
GP = {'Bahrain Grand Prix': 'Bahrain_ALO', 'Belgian Grand Prix': 'Belgian_GAS', 'Dutch Grand Prix': 'Dutch_STR', 'Hungarian Grand Prix': 'Hungarian_TSU',
      'Miami Grand Prix': 'Miami_OCO', 'Saudi Arabian Grand Prix': 'Saudi_Arabian_NOR', 'Singapore Grand Prix': 'Singapore_HUL', 'United States Grand Prix': 'United_States_MAG'}
NL = dict(zip(GP, [57, 44, 72, 70, 57, 50, 62, 56]))
S = pd.read_csv('data/tyre_sets_2024_reg_v10eGQ_20261028.csv'); print(list(S.columns))
L = ModelLoader(os.environ['F1_SIMULATION_DIR']); rows = []
for gp, md in GP.items():
    m = L.load(gp); av = sorted(int(c[1:]) if isinstance(c, str) else c for c in m.compounds)
    c = extract_simplified_calculator(m.laptime_clear, av); names = dict(zip(av, ['Hard', 'Medium', 'Soft']))
    b = {names[k]: [c.intercept + c.compound_effects[k], c.tyre_degradation[k]] for k in av}; ts = c.pit_in_cost + c.pit_out_cost
    z = S[(S.GP == gp) & (S.Driver == 'Driver_' + md.split('_')[-1])]
    sets = {k: sorted(z[z.Compound == k.upper()].Age.astype(float).tolist()) for k in ('Hard', 'Medium', 'Soft')}; N = NL[gp]
    left = {k: list(v) for k, v in sets.items()}; left['Medium'] = left['Medium'][1:]
    def mean_t(f, k=20):
        t = []
        for _ in range(k):
            t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
        return np.median(t) * 1e3
    st = mean_t(lambda: solve_MIP_GUROBI_start(b, ts, N, 0, [], 3, 0, N, verbose=False, set_ages=sets))
    per = [mean_t(lambda n=n: solve_MIP_GUROBI(b, ts, n, 'Medium', 10.0, ['Medium'], 3, 0, N, verbose=False, set_ages=left), 5) for n in range(N - 1, 0, -6)]
    pol = PPO.load(f"{os.environ['F1_RL_AGENTS_DIR']}/{md}/PPO/best_model.zip", device='cpu')
    obs = pol.observation_space.sample()
    inf = mean_t(lambda: pol.predict(obs, deterministic=True), 200)
    rows.append(dict(gp=gp, N=N, qmip_start_ms=round(st, 1), qmip_lap_ms=round(np.mean(per), 1), qmip_race_s=round(np.mean(per) * (N - 1) / 1e3 + st / 1e3, 2), ppo_decision_ms=round(inf, 2)))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv('results/runtimes_20261103/solve_times_20261103.csv', index=False); print(T.describe().loc[['min', '50%', 'max']].round(2).to_string())
