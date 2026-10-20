"""Archive of the v8 models and simulators for Zenodo — 2026-10-20.

Collects every file needed by reproduce/MODELS_MANIFEST_v8_20261020.csv and reproduce/figures_v8_20261020.sh under
trained_models/: the selected v8 agents (multi- and single-driver), the v7 simulators, and every file those trees reach
through symbolic links (e.g. trained_models/simulation/... and simulation_yfv4_opcion3_20261007/...). Links are kept as
links (relative), so shared pickles are stored once; the script checks that every link target is in the archive.

    python reproduce/pack_models_v8_20261020.py --dry-run          # list, sizes and link check only
    python reproduce/pack_models_v8_20261020.py --out <file.tar.gz>
"""
import os, csv, sys, tarfile, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM = os.path.join(ROOT, 'trained_models')
MANIFEST = os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST_v8_20261020.csv')
TREES = ['simulation_v7_20261015', 'simulation_onedriver_v7_20261015', 'rl_agents_v8_20261018/selected']


def walk(rel, files, links):
    """Add rel (file, dir or link, relative to trained_models/) and everything it reaches."""
    p = os.path.join(TM, rel)
    if rel in files or rel in links:
        return
    if os.path.islink(p):
        tgt = os.path.normpath(os.path.join(os.path.dirname(rel), os.readlink(p)))
        assert not tgt.startswith('..'), f'link outside trained_models/: {rel} -> {os.readlink(p)}'
        links[rel] = tgt
        walk(tgt, files, links)
    elif os.path.isdir(p):
        for n in sorted(os.listdir(p)):
            if n != '.DS_Store':
                walk(os.path.join(rel, n), files, links)
    elif os.path.isfile(p):
        files[rel] = os.path.getsize(p)
    else:
        raise FileNotFoundError(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default='f1-pit-stop-drl-v8-models.tar.gz')
    a = ap.parse_args()
    files, links = {}, {}
    for t in TREES:
        walk(t, files, links)
    for r in csv.DictReader(open(MANIFEST)):
        if r['scope'] == 'single_driver':
            walk(r['model_path'], files, links)
    walk(os.path.relpath(MANIFEST, TM), files, links) if MANIFEST.startswith(TM) else None
    # every link must resolve to something inside the archive
    def resolved(rel, seen=0):
        return resolved(links[rel], seen + 1) if rel in links else rel
    bad = [l for l in links if resolved(l) not in files and not any(f.startswith(resolved(l) + '/') for f in files)]
    tops = sorted({f.split('/')[0] for f in files})
    print(f'{len(files)} files, {sum(files.values()) / 1e6:.1f} MB, {len(links)} links; top-level trees: {tops}')
    print('unresolved links:', bad or 'none')
    if bad:
        sys.exit(1)
    if a.dry_run:
        return
    with tarfile.open(a.out, 'w:gz') as tar:
        for rel in sorted(set(files) | set(links)):
            tar.add(os.path.join(TM, rel), arcname=os.path.join('trained_models', rel), recursive=False)
        tar.add(MANIFEST, arcname='MODELS_MANIFEST_v8_20261020.csv')
    print('->', a.out, f'{os.path.getsize(a.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
