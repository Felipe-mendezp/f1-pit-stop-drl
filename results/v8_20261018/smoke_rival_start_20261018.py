"""2026-10-18 — Prueba local de los compuestos de partida Dirichlet-multinomial (v8). Sin entrenamiento.

    python scripts/rival_start_v8_20261018.py results/v8_20261018/smoke_rival_start_20261018.py [N]

(1) Sin alpha: los compuestos de partida coinciden con el codigo anterior (copia de respaldo del entorno), mismas semillas.
(2) Con alpha: por GP, N resets -> marginal por compuesto, proporcion media del mayoritario, P(19 iguales),
    comparado con las carreras reales 2021-23 (data/rival_start_alpha_2021_2023_20261018.json) y con alpha -> inf.
(3) Con compuestos reales 2024 en todos los rivales (modo Fig. 7): mismos compuestos y mismo estado del generador
    con y sin alpha (la Fig. 7 no cambia).
"""
import importlib.util, json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from environment.f1_env_all_drivers import create_f1_env_all_drivers  # noqa: E402
from model_loader import ModelLoader  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
ALPHA = float(os.environ['F1_RIVAL_START_ALPHA'])
GPS = {'Bahrain Grand Prix': 'ALO', 'Saudi Arabian Grand Prix': 'NOR', 'Miami Grand Prix': 'OCO',
       'Hungarian Grand Prix': 'TSU', 'Belgian Grand Prix': 'GAS', 'Dutch Grand Prix': 'STR',
       'Singapore Grand Prix': 'HUL', 'United States Grand Prix': 'MAG'}
BAK = os.environ.get('F1_ENV_BAK')
loader = ModelLoader()


def comps(env, seed):
    env.reset(seed=seed)
    return [d.compound for d in env.rival_drivers]


def make(gp, alpha):
    env = create_f1_env_all_drivers(gp=gp, driver='Driver_' + GPS[gp], deterministic=False, yf_enabled=True,
                                    loader=loader, verbose=False)
    env.rival_start_alpha = alpha
    return env


# (1)
if BAK:
    spec = importlib.util.spec_from_file_location('f1env_bak', BAK)
    bak = importlib.util.module_from_spec(spec); spec.loader.exec_module(bak)
    ok = True
    for gp in GPS:
        e_new = make(gp, None)
        e_old = bak.create_f1_env_all_drivers(gp=gp, driver='Driver_' + GPS[gp], deterministic=False, yf_enabled=True,
                                              loader=loader, verbose=False)
        e_old.rival_dnf = e_new.rival_dnf
        for s in range(50):
            ok &= comps(e_new, s) == comps(e_old, s)
            ok &= e_new.np_random.bit_generator.state == e_old.np_random.bit_generator.state
    print(f'(1) sin alpha == codigo anterior (8 GP x 50 semillas, compuestos y estado del RNG): {ok}')

# (2)
real = json.load(open(os.path.join(ROOT, 'data', 'rival_start_alpha_2021_2023_20261018.json')))
print(f'(2) alpha = {ALPHA}; real 2021-23: proporcion media del mayoritario = {real["mean_majority_share_real"]:.3f}')
print(f'{"GP":26s} {"compuestos":14s} {"marginal":20s} {"mayoritario":>11s} {"P(19 =)":>8s} {"P(>=15)":>8s} | uniforme: mayor. P(>=15)')
agg = []
for gp in GPS:
    res = {}
    for a in (ALPHA, None):
        e = make(gp, a)
        av = list(e.available_compounds)
        C = np.array([[av.index(c) for c in comps(e, 10_000 + s)] for s in range(N)])
        cnt = np.stack([(C == k).sum(1) for k in range(len(av))], 1)
        res[a] = (av, cnt.sum(0) / cnt.sum(), cnt.max(1) / cnt.shape[0] if False else cnt.max(1) / C.shape[1])
        if a:
            agg.append(cnt.max(1) / C.shape[1])
    av, marg, mx = res[ALPHA]; _, _, mxu = res[None]
    print(f'{gp:26s} {str(av):14s} {str(marg.round(3)):20s} {mx.mean():11.3f} {(mx == 1).mean():8.3f} '
          f'{(mx >= 15 / 19).mean():8.3f} | {mxu.mean():.3f} {(mxu >= 15 / 19).mean():.3f}')
agg = np.concatenate(agg)
print(f'{"todos":26s} {"":14s} {"":20s} {agg.mean():11.3f} {(agg == 1).mean():8.3f} {(agg >= 15 / 19).mean():8.3f}')
X = np.array([[r['SOFT'], r['MEDIUM'], r['HARD']] for r in real['races']])
sh = X.max(1) / X.sum(1)
print(f'{"real 2021-23 (19 carreras)":26s} {"":14s} {"":20s} {sh.mean():11.3f} {(sh == 1).mean():8.3f} {(sh >= 15 / 19).mean():8.3f}')

# (3)
ok = True
for gp in GPS:
    e1, e2 = make(gp, ALPHA), make(gp, None)
    av = list(e1.available_compounds)
    rng = np.random.default_rng(0)
    names = [d.name for d in (e1.reset(seed=0) and e1.rival_drivers)]
    real_c = {n: av[int(rng.integers(len(av)))] for n in names}
    for e in (e1, e2):
        e.eval_initial_compounds = dict(real_c)
    for s in range(20):
        ok &= comps(e1, s) == comps(e2, s) == [real_c[n] for n in [d.name for d in e1.rival_drivers]]
        ok &= e1.np_random.bit_generator.state == e2.np_random.bit_generator.state
print(f'(3) con compuestos reales en todos los rivales, alpha no cambia nada: {ok}')
