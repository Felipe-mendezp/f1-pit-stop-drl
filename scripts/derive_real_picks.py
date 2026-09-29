import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Derive real_picks.json / realyf_picks.json from an already-run decision-support
sweep: the prefix pit laps are model-dependent (deterministic policy under the
real 2024 events), so after retraining they must be re-extracted rather than
reused from the reference repo.

For each circuit it reads the LARGEST-lap `<md.lower()><suffix>_lapN_meta.json`
and takes the pit laps from its 'history' trace ([(lap, compound, pitted)]).
Fallback for metas without 'history': laps where agent tyrelife==0 at the next
lap's capture.

Usage:
    python scripts/derive_real_picks.py --suffix _real    # -> real_picks.json
    python scripts/derive_real_picks.py --suffix _realyf  # -> realyf_picks.json
"""

import argparse
import glob
import json
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Overridable so alternative sweeps (e.g. the 2026-08-04 fix-a/b regeneration)
# can derive picks from their own results dir without touching the legacy one.
RD = os.environ.get('F1_DS_RESULTS_DIR',
                    os.path.join(ROOT, 'results', 'decision_support'))

CIRCUITS = [
    ('Belgian Grand Prix', 'Driver_NOR', 'Belgian_NOR'),
    ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO'),
    ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR'),
    ('Emilia Romagna Grand Prix', 'Driver_TSU', 'Emilia_Romagna_TSU'),
    ('Hungarian Grand Prix', 'Driver_SAI', 'Hungarian_SAI'),
    ('Dutch Grand Prix', 'Driver_RUS', 'Dutch_RUS'),
    ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL'),
    ('United States Grand Prix', 'Driver_PIA', 'United_States_PIA'),
    ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO'),
]


def pit_laps_for(md: str, suffix: str):
    pfx = md.lower()
    metas = glob.glob(os.path.join(RD, f'{pfx}{suffix}_lap*_meta.json'))
    if not metas:
        return None, None
    def lapnum(p):
        m = re.search(r'_lap(\d+)_meta\.json$', p)
        return int(m.group(1)) if m else -1
    best = max(metas, key=lapnum)
    meta = json.load(open(best))
    n_laps = int(meta['n_laps'])
    hist = meta.get('history')
    if hist:
        # lap 0 "pit" is the initial compound choice, not a pit stop
        pits = [int(l) for l, _c, p in hist if p and int(l) >= 1]
    else:
        # Fallback: tyrelife==0 at capture of lap n => pit at lap n-1.
        pits = []
        for mp in sorted(metas, key=lapnum):
            m = json.load(open(mp))
            n = int(m['lap'])
            if n >= 2 and m['agent_state_at_lap']['tyrelife'] == 0:
                pits.append(n - 1)
    return sorted(set(pits)), n_laps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', required=True, choices=['_real', '_realyf'])
    args = ap.parse_args()

    picks = []
    for gp, drv, md in CIRCUITS:
        pits, n_laps = pit_laps_for(md, args.suffix)
        if pits is None:
            print(f'{md}: sin metas {args.suffix} — omitido')
            continue
        print(f'{md}: n_laps={n_laps} pit_laps={pits}')
        picks.append({'model_dir': md, 'gp': gp, 'driver': drv,
                      'n_laps': n_laps, 'pit_laps': pits})

    out = os.path.join(RD, f'{args.suffix[1:]}_picks.json')
    with open(out, 'w') as f:
        json.dump(picks, f, indent=2)
    print(f'Wrote {len(picks)} picks -> {out}')


if __name__ == '__main__':
    main()
