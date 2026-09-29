import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
Orchestrator: regenerate the FULL decision-support figure set (paper Sec. 7 +
per-circuit appendix) with the retrained models:

  - Belgian GP (paper showcase, Figs. 8-9): laps 1..43, M=500
  - Other 7 circuits: laps 10..min(50, N-2), M=500
  - Miami: sample laps 15/20/25/30/35 (as in the original figure set)

All rollouts use --fast-env (numpy path, validated at init) and --legal-mask
(paper Sec. 7: the continuation policy is restricted to rule-feasible actions,
including the FORCED mandatory two-compound stop at the last feasible lap —
without it, No-pit/same-compound arms end in massive DSQ->P20 tails).

Since 2026-07-25 the continuation after the forced action is DETERMINISTIC by
default (see decision_support_rollouts.py), so the counterfactual measures the
very policy whose recommendation the figures highlight. Nothing to pass here.

Prefix seeds: per-circuit clean seeds are read from
results/decision_support/seed_picks.json when present (fallback: seed 0 with a
warning). NOTE: the shipped seed_picks.json was scanned with the REFERENCE
repo's environment; after the 2026-07 fixes the same seed can yield a
different prefix here. Re-scan with scan_prefix_seeds.py + pick_seeds.py
before regenerating the paper figures.

After the rollouts, per-lap histograms (plot_decision_support.py) and the
evolution plots (plot_decision_evolution.py) are generated under
figs/decision_support/. For the real-2024 variants use post_real_figures.py /
post_realyf_figures.py; for the stint figure (Fig. 9) use
plot_decision_stints.py.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROLLOUTS = os.path.join(HERE, 'decision_support_rollouts.py')
PLOT_HIST = os.path.join(HERE, 'plot_decision_support.py')
PLOT_EVO = os.path.join(HERE, 'plot_decision_evolution.py')
PY = sys.executable

M = 500
MAXPROC = 10

CIRCUITS = [
    # (gp, driver, model_dir, N, laps)
    ('Belgian Grand Prix', 'Driver_NOR', 'Belgian_NOR', 44, list(range(1, 44))),
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO', 57, list(range(10, 51))),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR', 50, list(range(10, 49))),
    ('Emilia Romagna Grand Prix', 'Driver_TSU', 'Emilia_Romagna_TSU', 63, list(range(10, 51))),
    ('Hungarian Grand Prix', 'Driver_SAI', 'Hungarian_SAI', 70, list(range(10, 51))),
    ('Dutch Grand Prix', 'Driver_RUS', 'Dutch_RUS', 72, list(range(10, 51))),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL', 62, list(range(10, 51))),
    ('United States Grand Prix', 'Driver_PIA', 'United_States_PIA', 56, list(range(10, 51))),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO', 57, [15, 20, 25, 30, 35]),
]


def build_chunks():
    """Split each circuit's laps into contiguous chunks balanced by
    rollout cost (~ sum over laps of remaining race length N - lap)."""
    chunks = []
    for gp, drv, md, N, laps in CIRCUITS:
        total = sum(N - x for x in laps)
        n_chunks = 3 if total > 800 else (2 if total > 300 else 1)
        target = total / n_chunks
        made = 0
        cur, acc = [], 0.0
        for x in laps:
            cur.append(x)
            acc += N - x
            if acc >= target and made < n_chunks - 1:
                chunks.append((md, gp, drv, list(cur)))
                made += 1
                cur, acc = [], 0.0
        if cur:
            chunks.append((md, gp, drv, list(cur)))
    return chunks


def _load_seed_picks():
    """Per-circuit clean prefix seeds chosen by pick_seeds.py (see docstring)."""
    path = os.path.join(ROOT, 'results', 'decision_support', 'seed_picks.json')
    try:
        import json
        with open(path) as f:
            return {e['model_dir']: int(e['seed']) for e in json.load(f)}
    except (OSError, ValueError, KeyError):
        print(f"WARNING: could not read {path}; using prefix seed 0 everywhere.")
        return {}


SEED_PICKS = _load_seed_picks()


def run_chunk(md, gp, drv, laps):
    seed = SEED_PICKS.get(md, 0)
    cmd = [PY, ROLLOUTS, '--gp', gp, '--driver', drv, '--model-dir', md,
           '--M', str(M), '--fast-env', '--legal-mask',
           '--prefix-seed', str(seed), '--laps'] + [str(x) for x in laps]
    log = os.path.join(ROOT, 'logs', f'ds_{md}_{laps[0]}_{laps[-1]}.log')
    with open(log, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return md, laps[0], laps[-1], r.returncode


def plot_all():
    results_dir = os.path.join(ROOT, 'results', 'decision_support')
    figs_root = os.path.join(ROOT, 'figs', 'decision_support')
    failures = 0
    for gp, drv, md, N, laps in CIRCUITS:
        pfx = md.lower()
        png_dir = os.path.join(figs_root, 'png', md)
        pdf_dir = os.path.join(figs_root, 'pdf', md)
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)
        for n in laps:
            tag = f'{pfx}_lap{n}'
            if not os.path.exists(os.path.join(results_dir, f'{tag}.csv')):
                print(f'  missing rollout CSV for {tag}, skipping plot')
                failures += 1
                continue
            r = subprocess.run(
                [PY, PLOT_HIST, '--tag', tag,
                 '--out-png', os.path.join(png_dir, f'decision_support_lap{n}.png'),
                 '--out-pdf', os.path.join(pdf_dir, f'decision_support_lap{n}.pdf')],
                capture_output=True)
            if r.returncode != 0:
                failures += 1
                print(f'  histogram FAILED for {tag}: {r.stderr[-300:]}')
        # --drop-last-laps 1 matches post_real_figures.py / build_decision_
        # figures.sh: a pit on the sweep's last decision lap is a degenerate
        # edge case with artefactually good outcomes.
        r = subprocess.run(
            [PY, PLOT_EVO, '--model-dir', md,
             '--png-dir', png_dir, '--pdf-dir', pdf_dir,
             '--drop-last-laps', '1'],
            capture_output=True)
        if r.returncode != 0:
            failures += 1
            print(f'  evolution FAILED for {md}: {r.stderr[-300:]}')
        else:
            print(f'  plots done for {md}')
    return failures


if __name__ == '__main__':
    chunks = build_chunks()
    print(f'{len(chunks)} rollout chunks, MAXPROC={MAXPROC}')
    failures = 0
    with ThreadPoolExecutor(max_workers=MAXPROC) as ex:
        futs = {ex.submit(run_chunk, *c): c for c in chunks}
        for fut in as_completed(futs):
            md, lo, hi, rc = fut.result()
            status = 'OK' if rc == 0 else f'FAIL({rc})'
            failures += (rc != 0)
            print(f'[{status}] {md} laps {lo}-{hi}', flush=True)

    print('\n=== PLOTTING ===', flush=True)
    failures += plot_all()
    print(f'\nDONE. failures={failures}')
    sys.exit(1 if failures else 0)
