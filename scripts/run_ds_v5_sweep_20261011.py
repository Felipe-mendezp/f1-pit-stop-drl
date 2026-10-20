import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
2026-10-11 — Barrido de apoyo a la decision sobre v5 (simulador v4 opcion3 + coef. de piloto Ec. 2).

Mismo protocolo que scripts/run_ds_diag_sweep_20260930.py (referencia = agentes y simulador
publicados), pero con el simulador y los agentes de una opcion:
    F1_SIMULATION_DIR = trained_models/simulation_v5_20261010
    F1_RL_AGENTS_DIR  = trained_models/rl_agents_v5_20261010/selected
Correr tras source reproduce/env_paper_v3_20261007.sh (SC revelado, vueltas YF 2024 corregidas).
--world 2024: el entorno simula con neumaticos opcion3 2021-24 (scripts/ds_rollouts_world2024_20261011.py);
el agente sigue entrenado con 2021-23. Salida en results/decision_support_v5w24_20261011/.
2026-10-12: --option v5h (simulation_v5h8_20261012, agentes rl_agents_v5h_20261011/selected; requiere
F1_WORLD2024_TYRES=opcion3h) y --world belief (scripts/ds_rollouts_belief_20261012.py: carrera realizada en
el mundo 2024, proyecciones con la creencia 2021-23). Salida en results/decision_support_<option>w24<world>_20261012/.
8 GP del paper (sin Emilia Romagna) x {PPO, RecurrentPPO} x {--real-yf, --real-2024},
vueltas 1..N-1, --fast-env --legal-mask --sim-future --following-gap-scale 1.0.

Salida: results/decision_support_v5_20261011/<algo>/ y, con --figures, las figuras
por GP (derive_real_picks + post_real[yf]_figures) en .../figs/<algo>/{pdf,png}/<md>/.

    python scripts/run_ds_v5_sweep_20261011.py [--M 100] [--workers 6] [--figures]
"""

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLOUTS = os.path.join(ROOT, 'scripts', 'decision_support_rollouts.py')
CIRCUITS = [
    ('Belgian Grand Prix', 'Driver_NOR', 'Belgian_NOR', 44),
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO', 57),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR', 50),
    ('Hungarian Grand Prix', 'Driver_SAI', 'Hungarian_SAI', 70),
    ('Dutch Grand Prix', 'Driver_RUS', 'Dutch_RUS', 72),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL', 62),
    ('United States Grand Prix', 'Driver_PIA', 'United_States_PIA', 56),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO', 57),
]
ALGOS = ['PPO', 'RecurrentPPO']
VARIANTS = ['--real-yf', '--real-2024']


OPTIONS = {'v5': ('simulation_v5_20261010', 'rl_agents_v5_20261010'),
           'v5h': ('simulation_v5h8_20261012', 'rl_agents_v5h_20261011')}


def option_env(opt):
    sim, agents = OPTIONS[opt]
    return dict(os.environ, OMP_NUM_THREADS='1',
                F1_SIMULATION_DIR=os.path.join(ROOT, 'trained_models', sim),
                F1_RL_AGENTS_DIR=os.path.join(ROOT, 'trained_models', agents, 'selected'))


def run_job(job, M, out_root, env):
    gp, drv, md, N, algo, variant = job
    rd = os.path.join(out_root, algo)
    log = os.path.join(rd, 'logs', f'{md}{"_realyf" if variant == "--real-yf" else "_real"}.log')
    cmd = ['nice', '-n', '10', sys.executable, ROLLOUTS,
           '--gp', gp, '--driver', drv, '--model-dir', md, '--algo', algo,
           '--M', str(M), '--fast-env', '--legal-mask', variant, '--sim-future',
           '--following-gap-scale', '1.0', '--out-dir', rd,
           '--laps'] + [str(x) for x in range(1, N)]
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return job, rc, time.time() - t0


def figures(out_root, env, algos=ALGOS):
    for algo in algos:
        rd = os.path.join(out_root, algo)
        e = dict(env, F1_DS_WORLD2024='1' if 'w24' in out_root else '0', F1_DS_RESULTS_DIR=rd, F1_DS_FIG_DIR=os.path.join(out_root, 'figs', algo),
                 F1_DS_ALGO=algo, F1_DS_SIM_FUTURE='1', F1_DS_FOLLOWING_GAP_SCALE='1.0')
        for suffix, post in [('_real', 'post_real_figures.py'), ('_realyf', 'post_realyf_figures.py')]:
            subprocess.call([sys.executable, os.path.join(ROOT, 'scripts', 'derive_real_picks.py'),
                             '--suffix', suffix], cwd=ROOT, env=e)
            with open(os.path.join(rd, 'logs', f'figures{suffix}.log'), 'w') as f:
                rc = subprocess.call([sys.executable, os.path.join(ROOT, 'scripts', post)],
                                     cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
            print(f'figures {algo} {suffix}: rc={rc}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--option', default='v5', choices=list(OPTIONS))
    ap.add_argument('--world', default='none', choices=['none', '2024', 'belief'])
    ap.add_argument('--algos', nargs='+', default=ALGOS, choices=ALGOS)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--figures', action='store_true', help='also render the per-GP figures')
    ap.add_argument('--figures-only', action='store_true')
    args = ap.parse_args()
    global ROLLOUTS
    if args.world == '2024':
        ROLLOUTS = os.path.join(ROOT, 'scripts', 'ds_rollouts_world2024_20261011.py')
    elif args.world == 'belief':
        ROLLOUTS = os.path.join(ROOT, 'scripts', 'ds_rollouts_belief_20261012.py')
    if args.option == 'v5':
        assert args.world != 'belief'
        out_root = os.path.join(ROOT, 'results', 'decision_support_v5' + ('w24' if args.world == '2024' else '') + '_20261011')
    else:
        assert os.environ.get('F1_WORLD2024_TYRES') == 'opcion3h', 'F1_WORLD2024_TYRES=opcion3h'
        out_root = os.path.join(ROOT, 'results', f'decision_support_{args.option}' +
                                {'none': '', '2024': 'w24', 'belief': 'w24belief'}[args.world] + '_20261012')
    assert os.environ.get('F1_SC_REVEAL') == '1' and os.environ.get('F1_YF_LAPS_2024'), 'source reproduce/env_paper_v3_20261007.sh'
    env = option_env(args.option)
    for k in ('F1_SIMULATION_DIR', 'F1_RL_AGENTS_DIR'):
        assert os.path.isdir(env[k]), env[k]

    if not args.figures_only:
        for algo in args.algos:
            rd = os.path.join(out_root, algo)
            os.makedirs(os.path.join(rd, 'logs'), exist_ok=True)
            shutil.copy(os.path.join(ROOT, 'results', 'decision_support', 'initial_compounds.json'), rd)
        jobs = [(gp, drv, md, N, algo, v) for gp, drv, md, N in CIRCUITS for algo in args.algos for v in VARIANTS]
        jobs.sort(key=lambda j: -j[3] * (j[3] - 1))  # longest races first
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_job, j, args.M, out_root, env) for j in jobs]
            for k, fu in enumerate(as_completed(futs), 1):
                (gp, drv, md, N, algo, v), rc, dt = fu.result()
                print(f'[{k}/{len(jobs)}] {algo:12s} {md:20s} {v:12s} rc={rc} '
                      f'{dt / 60:.1f} min (elapsed {(time.time() - t0) / 60:.1f} min)', flush=True)
    if args.figures or args.figures_only:
        figures(out_root, env, args.algos)


if __name__ == '__main__':
    main()
