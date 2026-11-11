"""Archive of the look-ahead rollouts behind Figs. 8-11 for Zenodo (v10g) - 2026-11-05.

These CSV and JSON files are not versioned in git (see .gitignore) and take hours to recompute. They are the input of
Figs. 8-10 (results/decision_support_v10gw24belief_20261103/PPO) and Fig. 11 and Table L.1
(results/decision_support_v10gsmrealhist_20261102/PPO; Fig. 11 also verif_best_20261111/PPO). Run logs are left out. Extract it at the repository root.

    python reproduce/pack_results_v10g_20261105.py --dry-run
    python reproduce/pack_results_v10g_20261105.py --out <file.tar.gz>
"""
import argparse, hashlib, os, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES = ['results/decision_support_v10gw24belief_20261103/PPO', 'results/decision_support_v10gsmrealhist_20261102/PPO',
         'results/decision_support_v10gsmrealhist_20261102/verif_best_20261111/PPO']   # 2026-11-11: Fig. 11 (Bahrain)
KEEP = ('.csv', '.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default='f1-pit-stop-drl-v2.0.0-rollouts.tar.gz')
    a = ap.parse_args()
    files = []
    for t in TREES:
        tree = [os.path.relpath(os.path.join(d, n), ROOT) for d, _, fs in os.walk(os.path.join(ROOT, t)) for n in fs
                if n.endswith(KEEP)]
        assert tree, f'no files in {t}'
        print(f'{t}: {len(tree)} files, {sum(os.path.getsize(os.path.join(ROOT, f)) for f in tree) / 1e6:.1f} MB')
        files += tree
    if a.dry_run:
        return
    with tarfile.open(a.out, 'w:gz') as tar:
        for f in sorted(files):
            tar.add(os.path.join(ROOT, f), arcname=f, recursive=False)
    digest = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    print('->', a.out, f'{os.path.getsize(a.out) / 1e6:.1f} MB', 'sha256', digest)


if __name__ == '__main__':
    main()
