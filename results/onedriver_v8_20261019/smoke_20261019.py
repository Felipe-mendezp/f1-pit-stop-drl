"""2026-10-19 — Prueba local del entorno mono-piloto v8 (sin entrenamiento).

    F1_ENV_BAK=<copia del entorno anterior> python results/onedriver_v8_20261019/smoke_20261019.py

(1) Sin las variables: episodios (acciones al azar, incluida la accion 0 en la vuelta 0) identicos al codigo anterior
    (recompensas, observaciones, estado del generador).
(2) Con las variables (via scripts/one_driver_v8_20261019.py): la accion 0 en la vuelta 0 da el compuesto medio y el
    episodio es identico al de la accion 2 salvo la recompensa de la vuelta 0 (diferencia = -1000).
(3) Con las variables: episodios identicos al codigo anterior con las penalizaciones fijadas a mano en la instancia
    (-200, -100, -150) para acciones 1-3 en la vuelta 0.
(4) Con las variables: recompensa por evento = la del codigo anterior + (nuevo - viejo): parada en vuelta 1-5,
    paradas seguidas, parada en la ultima vuelta, sin cambio de compuesto (-1000 en ambos).
"""
import importlib.util, os, pickle, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TREE = os.path.join(ROOT, 'trained_models', 'simulation_onedriver_v7_20261015')
os.environ['F1_SIMULATION_DIR'] = TREE
os.environ.setdefault('F1_ONE_DRIVER_DSQ_PENALTY', '-1000')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from plot_fig5_stints_one_driver import SELECTED_GP, LAPS_DD  # noqa: E402
from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized as NEW  # noqa: E402

spec = importlib.util.spec_from_file_location('f1env_one_bak', os.environ['F1_ENV_BAK'])
bak = importlib.util.module_from_spec(spec); spec.loader.exec_module(bak)
OLD = bak.F1EnvOneDriverV2Optimized
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
V8 = ['F1_ONE_DRIVER_LAP0_ACTION0_MEDIUM', 'F1_ONE_DRIVER_LAP0_PENALTY', 'F1_ONE_DRIVER_PENALTY_CONSECUTIVE_PIT',
      'F1_ONE_DRIVER_PENALTY_EARLY_PIT', 'F1_ONE_DRIVER_PENALTY_PIT_LAST_LAP']
NEP = 30


def make(cls, gp):
    pred = pickle.load(open(os.path.join(TREE, f'{gp}_fast_predictor.pkl'), 'rb'))
    return cls(n_laps=LAPS_DD[gp], available_compounds=SELECTED_GP[gp], gp=gp, driver='VER', model_laptime=pred)


def flat(o):
    return np.array([o[k] for k in sorted(o)], dtype=float)


def episode(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    O, R, done, t = [flat(obs)], [], False, 0
    while not done:
        obs, r, done, _, _ = env.step(int(actions[t])); t += 1
        O.append(flat(obs)); R.append(float(r))
    return dict(obs=np.array(O), rew=np.array(R), rng=env.np_random.bit_generator.state, comp0=env.initial_compound,
                time=env.total_time)


def same(a, b, skip_r0=False):
    ra, rb = (a['rew'][1:], b['rew'][1:]) if skip_r0 else (a['rew'], b['rew'])
    return np.array_equal(a['obs'], b['obs']) and np.array_equal(ra, rb) and a['rng'] == b['rng']


def plans(n_laps, rng, a0_choices):
    """Acciones al azar: vuelta 0 en a0_choices; paradas con prob. 0.08 (incluye vueltas 1-5, seguidas y ultima)."""
    out = []
    for _ in range(NEP):
        a = np.where(rng.random(n_laps + 2) < 0.08, rng.integers(1, 4, n_laps + 2), 0)
        a[0] = rng.choice(a0_choices)
        out.append(a)
    # casos fijos: sin parada (DSQ), parada en la vuelta 2, paradas seguidas, parada en la ultima vuelta
    for pit in ([], [2], [20, 21], [n_laps]):
        a = np.zeros(n_laps + 2, dtype=int); a[0] = 1
        for l in pit:
            a[l] = 3
        out.append(a)
    return out


for k in V8:
    os.environ.pop(k, None)
ok = {k: True for k in range(1, 5)}
rng = np.random.default_rng(0)
for gp in GPS:                                   # (1) sin variables
    n, o = make(NEW, gp), make(OLD, gp)
    for i, a in enumerate(plans(LAPS_DD[gp], rng, [0, 1, 2, 3])):
        ok[1] &= same(episode(n, i, a), episode(o, i, a))
print(f'(1) sin variables == codigo anterior ({len(GPS)} GP x {NEP + 4} episodios): {ok[1]}')

import one_driver_v8_20261019  # noqa: E402,F401  (fija las variables v8 con setdefault)
print('   variables:', {k: os.environ[k] for k in V8})
for gp in GPS:
    n = make(NEW, gp)
    o = make(OLD, gp)
    o._penalty_consecutive_pit, o._penalty_early_pit, o._penalty_pit_last_lap = -200.0, -100.0, -150.0
    o0 = make(OLD, gp)                           # penalizaciones viejas
    med = n.available_compounds[1]
    for i, a in enumerate(plans(LAPS_DD[gp], rng, [1, 2, 3])):
        e_n, e_o, e_0 = episode(n, i, a), episode(o, i, a), episode(o0, i, a)
        ok[3] &= same(e_n, e_o)                  # (3)
        # (4) diferencia por vuelta = suma de (nuevo - viejo) por evento
        exp = np.zeros(len(a) - 1)
        L = len(e_n['rew']) - 1                  # pasos de carrera: t = 1..L
        for t in range(1, L + 1):
            pit = a[t] > 0
            if pit and t in (1, 2, 3, 4, 5): exp[t] += -100 - (-15)
            if pit and a[t - 1] > 0 and t > 1: exp[t] += -200 - (-20)
            if pit and t == L: exp[t] += -150 - (-30)
        ok[4] &= np.allclose(e_n['rew'] - e_0['rew'], exp[:L + 1]) and np.array_equal(e_n['obs'], e_0['obs'])
        # (2)
        b0, b2 = a.copy(), a.copy(); b0[0], b2[0] = 0, 2
        m0, m2 = episode(n, i, b0), episode(n, i, b2)
        ok[2] &= (m0['comp0'] == med and m2['comp0'] == med and same(m0, m2, skip_r0=True)
                  and np.isclose(m0['rew'][0] - m2['rew'][0], -1000))
    d = episode(n, 0, np.r_[1, np.zeros(LAPS_DD[gp] + 1, dtype=int)])
    print(f'   {gp:26s} compuestos {n.available_compounds} medio C{med}; sin parada: ultima recompensa '
          f'{d["rew"][-1]:.1f} (incluye DSQ {n._penalty_no_compound_change})')
names = {1: 'sin variables == codigo anterior', 2: 'accion 0 == accion 2 (medio) salvo -1000',
         3: 'con variables == codigo anterior con penalizaciones -200/-100/-150',
         4: 'diferencias de recompensa por evento exactas'}
for k in names:
    print(f'({k}) {names[k]}: {ok[k]}')
print('TODO OK' if all(ok.values()) else 'FALLA')
