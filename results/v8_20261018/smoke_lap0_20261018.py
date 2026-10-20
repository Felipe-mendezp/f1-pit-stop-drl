"""2026-10-18 — Prueba local de la accion 0 en la vuelta 0 (v8). Sin entrenamiento.

    F1_ENV_BAK=<copia del entorno anterior> python scripts/rival_start_v8_20261018.py results/v8_20261018/smoke_lap0_20261018.py

(1) Sin la opcion: episodios completos identicos al codigo anterior (acciones 0-3 en la vuelta 0; recompensas,
    observaciones y estado del generador).
(2) Con la opcion: la accion 0 da el compuesto medio y la penalizacion F1_LAP0_PENALTY; el episodio es identico al de
    la accion 2 (mismo generador, mismas observaciones) salvo la recompensa de la vuelta 0 (diferencia = penalizacion).
(3) Con la opcion: las acciones 1-3 dan el mismo episodio que sin ella.
(4) Con la opcion, partir con el medio y no cambiar de compuesto sigue dando descalificacion.
(5) La factory lee F1_LAP0_ACTION0_MEDIUM y F1_LAP0_PENALTY.
"""
import importlib.util, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from environment.f1_env_all_drivers import create_f1_env_all_drivers  # noqa: E402
from model_loader import ModelLoader  # noqa: E402

GPS = {'Bahrain Grand Prix': 'ALO', 'Saudi Arabian Grand Prix': 'NOR', 'Miami Grand Prix': 'OCO',
       'Hungarian Grand Prix': 'TSU', 'Belgian Grand Prix': 'GAS', 'Dutch Grand Prix': 'STR',
       'Singapore Grand Prix': 'HUL', 'United States Grand Prix': 'MAG'}
PEN = float(os.environ['F1_LAP0_PENALTY'])
NSEED = 5
loader = ModelLoader()


def make(gp, mod=None, medium=False):
    f = (mod.create_f1_env_all_drivers if mod else create_f1_env_all_drivers)
    env = f(gp=gp, driver='Driver_' + GPS[gp], deterministic=False, yf_enabled=True, loader=loader, verbose=False)
    env.lap0_action0_medium = medium
    return env


def flat(obs):
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[k], dtype=float).ravel() for k in sorted(obs)])
    return flat(obs)


def episode(env, seed, a0, pit_lap=20, pit_action=3):
    """Accion a0 en la vuelta 0; parada con pit_action en pit_lap (None = nunca); 0 el resto."""
    obs, _ = env.reset(seed=seed)
    obs_l, rew, done, t = [flat(obs)], [], False, 0
    comp0 = None
    while not done:
        a = a0 if t == 0 else (pit_action if pit_lap is not None and t == pit_lap else 0)
        obs, r, term, trunc, _ = env.step(a)
        if t == 0:
            comp0 = env.agent_driver.initial_compound
        obs_l.append(flat(obs)); rew.append(float(r)); done = term or trunc; t += 1
    return dict(obs=np.array(obs_l), rew=np.array(rew), comp0=comp0, pos=env.agent_driver.position,
                rng=env.np_random.bit_generator.state, changed=env.agent_driver.has_changed_compound)


def same(a, b, skip_r0=False):
    r_a, r_b = (a['rew'][1:], b['rew'][1:]) if skip_r0 else (a['rew'], b['rew'])
    return (a['obs'].shape == b['obs'].shape and np.array_equal(a['obs'], b['obs']) and np.array_equal(r_a, r_b)
            and a['rng'] == b['rng'] and a['pos'] == b['pos'])


bak_path = os.environ['F1_ENV_BAK']
spec = importlib.util.spec_from_file_location('f1env_bak', bak_path)
bak = importlib.util.module_from_spec(spec); spec.loader.exec_module(bak)
print(f'F1_LAP0_PENALTY = {PEN}; F1_RIVAL_START_ALPHA = {os.environ.get("F1_RIVAL_START_ALPHA")}; {NSEED} semillas por GP')

ok = {k: True for k in range(1, 6)}
for gp in GPS:
    e_new, e_old, e_med = make(gp), make(gp, bak), make(gp, medium=True)
    for e in (e_old,):
        e.rival_dnf, e.rival_start_alpha = e_new.rival_dnf, getattr(e_new, 'rival_start_alpha', None)
    for e in (e_med,):
        e.lap0_penalty = PEN
    comps = e_new.available_compounds
    med = comps[len(comps) // 2]
    n0 = 0
    for s in range(NSEED):
        # (1)
        for a0 in range(4):
            ok[1] &= same(episode(e_new, s, a0), episode(e_old, s, a0))
        # (2)
        m0, m2 = episode(e_med, s, 0), episode(e_med, s, 2)
        ok[2] &= (m0['comp0'] == med and m2['comp0'] == med and same(m0, m2, skip_r0=True)
                  and np.isclose(m0['rew'][0] - m2['rew'][0], PEN))
        # (3)
        for a0 in (1, 2, 3):
            ok[3] &= same(episode(e_med, s, a0), episode(e_new, s, a0))
        # (4)
        d = episode(e_med, s, 0, pit_lap=None)
        ok[4] &= (not d['changed']) and d['comp0'] == med
        n0 += 1
    print(f'  {gp:26s} compuestos {comps} -> medio C{med}: (1) {ok[1]}  (2) {ok[2]}  (3) {ok[3]}  (4) {ok[4]}')

# (4b) la penalizacion de descalificacion se aplica: retorno sin cambio vs con cambio de compuesto, ultima recompensa
e = make('Bahrain Grand Prix', medium=True); e.lap0_penalty = PEN
d, c = episode(e, 0, 0, pit_lap=None), episode(e, 0, 0)
print(f'(4b) Bahrain semilla 0, accion 0: ultima recompensa sin cambio de compuesto {d["rew"][-1]:.1f} '
      f'(DSQ {e._penalty_disqualification}); con parada a C{e.available_compounds[2]} en la vuelta 20: {c["rew"][-1]:.1f}')

# (5)
e = create_f1_env_all_drivers(gp='Dutch Grand Prix', driver='Driver_STR', deterministic=False, yf_enabled=True,
                              loader=loader, verbose=False)
ok[5] = getattr(e, 'lap0_action0_medium', False) is True and getattr(e, 'lap0_penalty', None) == PEN
e.reset(seed=0); _, r0, *_ = e.step(0)
print(f'(5) factory: lap0_action0_medium={getattr(e, "lap0_action0_medium", None)} lap0_penalty={getattr(e, "lap0_penalty", None)}'
      f' -> step(0): recompensa {r0}, compuesto C{e.agent_driver.initial_compound}')
names = {1: 'sin la opcion == codigo anterior', 2: 'accion 0 == accion 2 salvo la penalizacion', 3: 'acciones 1-3 sin cambio',
         4: 'medio sin cambio de compuesto -> sin cambio registrado (DSQ)', 5: 'factory lee las variables'}
for k in names:
    print(f'({k}) {names[k]}: {ok[k]}')
print('TODO OK' if all(ok.values()) else 'FALLA')
