import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Post-sweep figure driver for the REAL-2024 decision-support experiment.
For each pick in results/decision_support/real_picks.json (model_dir, gp, driver,
n_laps, pit_laps):

  1. add the '<Model_dir>_real' key to initial_compounds.json
  2. check_alignment on the full real-2024 data (report only)
  3. render the 3 evolution figures (--drop-last-laps 1)
  4. render the pit-lap histograms
  5. render the stint diagram at each pit lap in --real-2024 mode (the four bars
     share the actual race's SC/VSC + rivals, so no continuation-seed search is
     needed and event alignment is exact by construction).

Figures land in figs/decision_support_bundle/{pdf,png}/<Model_dir>/ with a
_real tag. Optional argv filters restrict to given model_dir names.
"""

import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Overridable (2026-08-04) so alternative sweeps (fix-a/b regeneration) render
# from their own results dir; default keeps the legacy layout untouched.
RD = os.environ.get('F1_DS_RESULTS_DIR',
                    os.path.join(ROOT, 'results', 'decision_support'))
# Figures stay INSIDE the submission repo (do not touch the original
# ../Code/Figures tree). Override with F1_DS_FIG_DIR if needed.
FIG = os.environ.get('F1_DS_FIG_DIR',
                     os.path.join(ROOT, 'figs', 'decision_support_bundle'))
ENV = dict(os.environ, OMP_NUM_THREADS='1')
STINT_SEED = 1000  # lap-time-noise seed for the representative real continuation
# Env-physics flags forwarded to plot_decision_stints.py so the stint
# continuations run under the SAME physics as the sweep CSVs (2026-08-04).
STINT_PHYS = []
if os.environ.get('F1_DS_PIT_DET_PASS'):
    STINT_PHYS += ['--pit-det-pass']
if os.environ.get('F1_DS_FOLLOWING_GAP_SCALE'):
    STINT_PHYS += ['--following-gap-scale', os.environ['F1_DS_FOLLOWING_GAP_SCALE']]
if os.environ.get('F1_DS_SIM_FUTURE'):
    STINT_PHYS += ['--sim-future']


def sh(args, capture=False):
    if capture:
        r = subprocess.run(args, env=ENV, cwd=ROOT, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return r.returncode, r.stdout
    return subprocess.call(args, env=ENV, cwd=ROOT), ''


def main():
    picks = json.load(open(os.path.join(RD, 'real_picks.json')))
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        picks = [p for p in picks if p['model_dir'] in only]
    init_path = os.path.join(RD, 'initial_compounds.json')
    init = json.load(open(init_path))
    summary = []

    for p in picks:
        md, gp, drv = p['model_dir'], p['gp'], p['driver']
        pfx = md.lower(); tag = f"{pfx}_real"
        pdf = os.path.join(FIG, 'pdf', md); png = os.path.join(FIG, 'png', md)
        os.makedirs(pdf, exist_ok=True); os.makedirs(png, exist_ok=True)
        print(f"\n########## {md} (REAL 2024) ##########", flush=True)

        # 1. initial-compound key
        meta1 = json.load(open(os.path.join(RD, f"{tag}_lap1_meta.json")))
        comp = meta1['agent_state_at_lap']['compound']
        comps = meta1['available_compounds']
        action = comps.index(comp) + 1
        init[f"{md}_real"] = {'compound': comp, 'action': action,
                              'label': meta1['action_labels'][str(action)]}
        json.dump(init, open(init_path, 'w'), indent=2)

        # 2. alignment (report)
        rc, out = sh([sys.executable, 'scripts/check_alignment.py', '--tag', tag,
                      '--thresh', '0.05', '--results-dir', RD], capture=True)
        flag = [l for l in out.splitlines() if l.startswith('flagged')]
        print(flag[0] if flag else out[-300:])
        summary.append((md, flag[0] if flag else 'alignment ERROR'))

        # 3. evolution figures
        rc, out = sh([sys.executable, 'scripts/plot_decision_evolution.py',
                      '--model-dir', f'{md}_real', '--pdf-dir', pdf,
                      '--png-dir', png, '--name', tag, '--drop-last-laps', '1',
                      '--results-dir', RD],
                     capture=True)
        print('evolution: ' + ('ok' if rc == 0 else f'FAIL\n{out[-600:]}'))

        # 4. pit-lap histograms
        for n in p['pit_laps']:
            rc, out = sh([sys.executable, 'scripts/plot_decision_support.py',
                          '--tag', f'{tag}_lap{n}', '--results-dir', RD,
                          '--out-pdf', os.path.join(pdf, f'decision_support_real_lap{n}.pdf'),
                          '--out-png', os.path.join(png, f'decision_support_real_lap{n}.png')],
                         capture=True)
            print(f'histogram lap {n}: ' + ('ok' if rc == 0 else f'FAIL\n{out[-600:]}'))

        # 5. stint diagrams (real events, no seed search)
        for n in p['pit_laps']:
            rc, out = sh([sys.executable, 'scripts/plot_decision_stints.py',
                          '--gp', gp, '--driver', drv, '--model-dir', md,
                          '--lap', str(n), '--real-2024', '--seed', str(STINT_SEED),
                          '--results-dir', RD, *STINT_PHYS,
                          '--out-pdf', os.path.join(pdf, f'decision_stints_real_lap{n}.pdf'),
                          '--out-png', os.path.join(png, f'decision_stints_real_lap{n}.png')],
                         capture=True)
            print(f'stints lap {n}: ' + ('ok' if rc == 0 else f'FAIL\n{out[-600:]}'))

    print('\n\n========== SUMMARY ==========')
    for md, msg in summary:
        print(f"{md}: {msg}")


if __name__ == '__main__':
    main()
