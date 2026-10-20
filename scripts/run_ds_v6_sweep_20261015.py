"""
2026-10-15 — Barrido de apoyo a la decision sobre v6 (8 GP, pilotos v6), para contrastar con v7.

Mismo protocolo que scripts/run_ds_v5_sweep_20261011.py --option v5h --world belief --M 500 --figures (carrera hasta la
vuelta n en el mundo 2024 opcion3h, proyecciones con la creencia 2021-23; PPO y RecurrentPPO; variantes --real-yf y
--real-2024; vueltas 1..N-1; --fast-env --legal-mask --sim-future --following-gap-scale 1.0), con:
    F1_SIMULATION_DIR = trained_models/simulation_v6_20261014
    F1_RL_AGENTS_DIR  = trained_models/rl_agents_v6_20261014/selected
    pilotos v6 (Belgian GAS, Hungarian TSU, Dutch STR, US MAG; los otros 4 igual que v5h): cada rollout y
    plot_decision_stints corren a traves de scripts/gp_drivers_v5hd_20261014.py (GP_CONFIGS en memoria).
Rivales: en --real-2024, estrategias reales 2024 (incluye los abandonos reales, como la Fig. 7); en --real-yf, rivales
estocasticos sin abandonos (como el barrido v5h). No modifica los scripts originales: derive_real_picks.py y
post_real[yf]_figures.py se ejecutan con su lista de pilotos / llamada a plot_decision_stints reemplazadas en memoria.
Salida: results/decision_support_v6w24belief_20261015/<algo>/ y figs/.
    source reproduce/env_paper_v3_20261007.sh
    F1_WORLD2024_TYRES=opcion3h python scripts/run_ds_v6_sweep_20261015.py [--M 500] [--workers 8] [--figures]
--version v7 (2026-10-15): simulation_v7_20261015, rl_agents_v7_20261015/selected, mundo 2024 v7
(F1_WORLD2024_TYRES=v7), salida results/decision_support_v7w24belief_20261015/. Mismos pilotos y protocolo.
    F1_WORLD2024_TYRES=v7 python scripts/run_ds_v6_sweep_20261015.py --version v7 --M 500 --workers 8 --figures
--version v8 (2026-10-19): simulador y mundo 2024 de v7, agentes rl_agents_v8_20261018/selected (reglas 5/7/8 +
--max-dsq 0.01), salida results/decision_support_v8w24belief_20261019/. Requiere las variables v8 del entorno
(F1_RIVAL_START_ALPHA, F1_LAP0_ACTION0_MEDIUM, F1_LAP0_PENALTY), que lee create_f1_env_all_drivers.
"""
import argparse, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAP = os.path.join(ROOT, 'scripts', 'gp_drivers_v5hd_20261014.py')
ROLLOUTS = os.path.join(ROOT, 'scripts', 'ds_rollouts_belief_20261012.py')
OUT = os.path.join(ROOT, 'results', 'decision_support_v6w24belief_20261015')
SIM = os.path.join(ROOT, 'trained_models', 'simulation_v6_20261014')
AGENTS = os.path.join(ROOT, 'trained_models', 'rl_agents_v6_20261014', 'selected')
CIRCUITS = [
    ('Belgian Grand Prix', 'Driver_GAS', 'Belgian_GAS', 44),
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO', 57),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR', 50),
    ('Hungarian Grand Prix', 'Driver_TSU', 'Hungarian_TSU', 70),
    ('Dutch Grand Prix', 'Driver_STR', 'Dutch_STR', 72),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL', 62),
    ('United States Grand Prix', 'Driver_MAG', 'United_States_MAG', 56),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO', 57),
]
ALGOS = ['PPO', 'RecurrentPPO']
VERSIONS = {'v6': ('simulation_v6_20261014', 'rl_agents_v6_20261014', 'decision_support_v6w24belief_20261015', 'opcion3h'),
            'v7': ('simulation_v7_20261015', 'rl_agents_v7_20261015', 'decision_support_v7w24belief_20261015', 'v7'),
            'v8': ('simulation_v7_20261015', 'rl_agents_v8_20261018', 'decision_support_v8w24belief_20261019', 'v7')}
VARIANTS = ['--real-yf', '--real-2024']


def env_base():
    return dict(os.environ, OMP_NUM_THREADS='1', F1_SIMULATION_DIR=SIM, F1_RL_AGENTS_DIR=AGENTS)


def run_job(job, M, env):
    gp, drv, md, N, algo, variant = job
    rd = os.path.join(OUT, algo)
    log = os.path.join(rd, 'logs', f'{md}{"_realyf" if variant == "--real-yf" else "_real"}.log')
    cmd = ['nice', '-n', '10', sys.executable, WRAP, ROLLOUTS,
           '--gp', gp, '--driver', drv, '--model-dir', md, '--algo', algo,
           '--M', str(M), '--fast-env', '--legal-mask', variant, '--sim-future',
           '--following-gap-scale', '1.0', '--out-dir', rd, '--laps'] + [str(x) for x in range(1, N)]
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return job, rc, time.time() - t0


def _exec_patched(path, repl):
    src = open(path).read()
    for a, b in repl:
        assert src.count(a) >= 1, (path, a)
        src = src.replace(a, b)
    sys.argv[0] = path
    exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})


def derive(suffix):
    """derive_real_picks.py con los pilotos v6 (se ejecuta en un subproceso con F1_DS_RESULTS_DIR)."""
    path = os.path.join(ROOT, 'scripts', 'derive_real_picks.py')
    old = open(path).read()
    i, j = old.index('CIRCUITS = ['), old.index(']\n', old.index('CIRCUITS = [')) + 2
    new = 'CIRCUITS = ' + repr([(g, d, m) for g, d, m, _ in CIRCUITS]) + '\n'
    sys.argv = [path, '--suffix', suffix]
    _exec_patched(path, [(old[i:j], new)])


def post(script):
    """post_real[yf]_figures.py con plot_decision_stints a traves del envoltorio de pilotos v6."""
    path = os.path.join(ROOT, 'scripts', script)
    sys.argv = [path]
    _exec_patched(path, [("[sys.executable, 'scripts/plot_decision_stints.py',",
                          f"[sys.executable, {WRAP!r}, 'scripts/plot_decision_stints.py',")])


def figures(env, algos):
    for algo in algos:
        rd = os.path.join(OUT, algo)
        e = dict(env, F1_DS_WORLD2024='1', F1_DS_RESULTS_DIR=rd, F1_DS_FIG_DIR=os.path.join(OUT, 'figs', algo),
                 F1_DS_ALGO=algo, F1_DS_SIM_FUTURE='1', F1_DS_FOLLOWING_GAP_SCALE='1.0')
        for suffix, script in [('_real', 'post_real_figures.py'), ('_realyf', 'post_realyf_figures.py')]:
            subprocess.call([sys.executable, __file__, '--derive', suffix], cwd=ROOT, env=e)
            with open(os.path.join(rd, 'logs', f'figures{suffix}.log'), 'w') as f:
                rc = subprocess.call([sys.executable, __file__, '--post', script], cwd=ROOT, env=e,
                                     stdout=f, stderr=subprocess.STDOUT)
            print(f'figures {algo} {suffix}: rc={rc}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algos', nargs='+', default=ALGOS, choices=ALGOS)
    ap.add_argument('--M', type=int, default=500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gps', nargs='+', default=None, help='model_dir (p. ej. Singapore_HUL); por defecto los 8')
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--version', default='v6', choices=list(VERSIONS))
    ap.add_argument('--derive', default=None)
    ap.add_argument('--post', default=None)
    a = ap.parse_args()
    if a.derive:
        return derive(a.derive)
    if a.post:
        return post(a.post)
    global SIM, AGENTS, OUT
    sim, agents, out, tyres = VERSIONS[a.version]
    SIM = os.path.join(ROOT, 'trained_models', sim)
    AGENTS = os.path.join(ROOT, 'trained_models', agents, 'selected')
    OUT = os.path.join(ROOT, 'results', out)
    assert os.environ.get('F1_WORLD2024_TYRES') == tyres, f'F1_WORLD2024_TYRES={tyres}'
    assert os.environ.get('F1_SC_REVEAL') == '1' and os.environ.get('F1_YF_LAPS_2024'), \
        'source reproduce/env_paper_v3_20261007.sh'
    if a.version == 'v8':   # 2026-10-19
        assert all(os.environ.get(k) for k in ('F1_RIVAL_START_ALPHA', 'F1_LAP0_ACTION0_MEDIUM', 'F1_LAP0_PENALTY')), \
            '--version v8: exportar F1_RIVAL_START_ALPHA, F1_LAP0_ACTION0_MEDIUM y F1_LAP0_PENALTY'
    assert 'F1_RIVAL_DNF' not in os.environ, 'sin F1_RIVAL_DNF (rivales --real-yf sin abandonos, como v5h)'
    for p in (SIM, AGENTS):
        assert os.path.isdir(p), p
    env = env_base()
    circuits = [c for c in CIRCUITS if a.gps is None or c[2] in a.gps]
    if not a.figures_only:
        for algo in a.algos:
            rd = os.path.join(OUT, algo)
            os.makedirs(os.path.join(rd, 'logs'), exist_ok=True)
            if not os.path.exists(os.path.join(rd, 'initial_compounds.json')):
                shutil.copy(os.path.join(ROOT, 'results', 'decision_support', 'initial_compounds.json'), rd)
        jobs = [(gp, drv, md, N, algo, v) for gp, drv, md, N in circuits for algo in a.algos for v in VARIANTS]
        jobs.sort(key=lambda j: -j[3] * (j[3] - 1))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_job, j, a.M, env) for j in jobs]
            for k, fu in enumerate(as_completed(futs), 1):
                (gp, drv, md, N, algo, v), rc, dt = fu.result()
                print(f'[{k}/{len(jobs)}] {algo:12s} {md:20s} {v:12s} rc={rc} {dt / 60:.1f} min '
                      f'(elapsed {(time.time() - t0) / 60:.1f} min)', flush=True)
    if a.figures or a.figures_only:
        figures(env, a.algos)


if __name__ == '__main__':
    main()
