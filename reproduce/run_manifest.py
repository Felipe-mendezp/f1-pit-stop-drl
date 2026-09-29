"""Retrain the paper's agents from reproduce/MODELS_MANIFEST.csv.

Each manifest row records how one shipped model (or one single-driver training call) was produced:
script, arguments, effective seed and the environment switches that were active. This runner
replays those rows into a NEW output tree (it never writes into trained_models/rl_agents*).

    python reproduce/run_manifest.py --scope multi_driver  --out trained_models/retrained_multi   [--dry-run] [--only Saudi]
    python reproduce/run_manifest.py --scope single_driver --out trained_models/retrained_single
    python reproduce/run_manifest.py --scope reward_comparison --out trained_models/retrained_reward

Budgets: 5M environment steps per model (multi-driver ~20-60 min per model on a laptop CPU,
single-driver ~45 min per GP for the 5 algorithms). Training is seeded; results are expected to
match the shipped models closely but bit-identity across machines/library versions is not guaranteed.
"""
import argparse
import csv
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'
BASE_ENV = {'OMP_NUM_THREADS': '1'}


def row_env(row, out):
    env = dict(os.environ)
    env.update(BASE_ENV)
    for k in ('F1_DSQ_PENALTY', 'F1_LASTLAP_PITOUT', 'F1_SC_BUNCHING', 'F1_SC_BUNCH_DELTA',
              'F1_FOLLOWING_GAP_SCALE', 'F1_ONE_DRIVER_DSQ_PENALTY', 'F1_RL_AGENTS_DIR', 'F1_RL_REWARD_DIR'):
        env.pop(k, None)  # start from a clean physics state; the row decides
    for tok in row['env'].split('(')[0].split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            v = v.replace('{OUT}', out)
            if v != '0':
                env[k] = v
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scope', required=True, choices=['multi_driver', 'single_driver', 'reward_comparison'])
    ap.add_argument('--out', required=True, help='new output tree (created)')
    ap.add_argument('--only', default=None, help='substring filter on model_path (e.g. Saudi, RecurrentPPO)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for protected in ('rl_agents', 'rl_agents_reward'):
        if os.path.realpath(out) == os.path.realpath(os.path.join(ROOT, 'trained_models', protected)):
            sys.exit(f'refusing to write into the shipped tree trained_models/{protected}')
    os.makedirs(out, exist_ok=True)
    py = os.environ.get('PYTHON', sys.executable)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST.csv')))
            if r['scope'] == a.scope and (a.only is None or a.only in r['model_path'])]
    print(f'{len(rows)} training call(s) -> {out}')
    for r in rows:
        args = r['args'].replace('{PHYS}', PHYS).replace('{OUT}', shlex.quote(out))
        cmd = [py, os.path.join(ROOT, r['script'])] + shlex.split(args)
        env = row_env(r, out)
        shown = ' '.join(f'{k}={env[k]}' for k in sorted(env) if k.startswith('F1_'))
        print(f"\n# {r['model_path']}  (seed {r['effective_seed']})  {r['notes']}\n{shown} {' '.join(shlex.quote(c) for c in cmd)}")
        if not a.dry_run:
            subprocess.run(cmd, env=env, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
