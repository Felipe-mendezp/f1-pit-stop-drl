import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Diagnostic decision-support sweep (2026-09-30): shipped agents, NO retraining.

For each of the 9 circuits x {PPO, RecurrentPPO} x {real_yf (i), real_2024 (ii)}
runs decision_support_rollouts.py over every decision lap 1..N-1 with the
paper protocol (--fast-env --legal-mask --sim-future --following-gap-scale 1.0).
The race realization (prefix) is the agent's own deterministic policy under the
variant's conditioning; continuations from lap n are fully simulated.

Results go to results/decision_support_diag_20260930/<algo>/ (one dir per
algorithm, since the output tags do not include the algorithm), each seeded
with a copy of results/decision_support/initial_compounds.json so that
derive_real_picks.py / post_real[yf]_figures.py can run on it afterwards.

Run after `source reproduce/env_paper.sh`. Jobs run under `nice` so other
work on the machine keeps priority.

Usage:
    python scripts/run_ds_diag_sweep_20260930.py [--M 100] [--workers 6]
"""

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLOUTS = os.path.join(ROOT, 'scripts', 'decision_support_rollouts.py')
OUT_ROOT = os.path.join(ROOT, 'results', 'decision_support_diag_20260930')

CIRCUITS = [
    # (gp, driver, model_dir, N)
    ('Belgian Grand Prix', 'Driver_NOR', 'Belgian_NOR', 44),
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO', 57),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR', 50),
    ('Emilia Romagna Grand Prix', 'Driver_TSU', 'Emilia_Romagna_TSU', 63),
    ('Hungarian Grand Prix', 'Driver_SAI', 'Hungarian_SAI', 70),
    ('Dutch Grand Prix', 'Driver_RUS', 'Dutch_RUS', 72),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL', 62),
    ('United States Grand Prix', 'Driver_PIA', 'United_States_PIA', 56),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO', 57),
]
ALGOS = ['PPO', 'RecurrentPPO']
VARIANTS = ['--real-yf', '--real-2024']


def run_job(job, M):
    gp, drv, md, N, algo, variant = job
    rd = os.path.join(OUT_ROOT, algo)
    log = os.path.join(rd, 'logs', f'{md}{"_realyf" if variant == "--real-yf" else "_real"}.log')
    cmd = ['nice', '-n', '10', sys.executable, ROLLOUTS,
           '--gp', gp, '--driver', drv, '--model-dir', md, '--algo', algo,
           '--M', str(M), '--fast-env', '--legal-mask', variant, '--sim-future',
           '--following-gap-scale', '1.0', '--out-dir', rd,
           '--laps'] + [str(x) for x in range(1, N)]
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS='1'))
    return job, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    for algo in ALGOS:
        rd = os.path.join(OUT_ROOT, algo)
        os.makedirs(os.path.join(rd, 'logs'), exist_ok=True)
        shutil.copy(os.path.join(ROOT, 'results', 'decision_support', 'initial_compounds.json'), rd)

    jobs = [(gp, drv, md, N, algo, v) for gp, drv, md, N in CIRCUITS
            for algo in ALGOS for v in VARIANTS]
    jobs.sort(key=lambda j: -j[3] * (j[3] - 1))  # longest races first

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, args.M) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            (gp, drv, md, N, algo, v), rc, dt = fu.result()
            print(f'[{k}/{len(jobs)}] {algo:12s} {md:20s} {v:12s} rc={rc} '
                  f'{dt / 60:.1f} min (elapsed {(time.time() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
