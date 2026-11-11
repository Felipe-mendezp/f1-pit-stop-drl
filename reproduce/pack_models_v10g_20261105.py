"""Archive of the current models and simulators for Zenodo (v10g) - 2026-11-05.

Same approach as pack_models_v8_20261020.py. Collects under trained_models/ what the manuscript's results load:
  - simulators: simulation_v10eGQ_20261028, simulation_onedriver_v10eGQ_20261028, plus every file they reach through
    symbolic links (config.json and others in simulation/ and simulation_v6_20261014/);
  - multi-driver agents: retrained_v10g_20261030/selected (40; Figs. 6-10);
  - further-trained agents: retrained_v10g_startmix_20261101/selected (8; Fig. 11);
  - single-driver agents: the targets of the links in results/onedriver_v10eGQ_20261028/selected (40; Fig. 5, Table N.1).
Links are kept as relative links, so shared pickles are stored once, and every link target must be in the archive.
Non-selected seeds, training logs and earlier model trees are left out. The archive also carries a generated manifest
with the seed and SHA-256 of each agent. Extract it at the repository root.

    python reproduce/pack_models_v10g_20261105.py --dry-run          # list, sizes and checks only
    python reproduce/pack_models_v10g_20261105.py --out <file.tar.gz>
"""
import argparse, csv, glob, hashlib, io, os, re, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM = os.path.join(ROOT, 'trained_models')
ONE_DIR = os.path.join(ROOT, 'results', 'onedriver_v10eGQ_20261028')
ONE_SEL = os.path.join(ONE_DIR, 'selected')
MULTI = {'multi_driver': 'retrained_v10g_20261030', 'further_training': 'retrained_v10g_startmix_20261101'}
TREES = ['simulation_v10eGQ_20261028', 'simulation_onedriver_v10eGQ_20261028'] + [f'{t}/selected' for t in MULTI.values()]
MANIFEST_NAME = 'MODELS_MANIFEST_v10g_20261105.csv'
EXPECTED_ZIPS = {'multi_driver': 40, 'further_training': 8, 'single_driver': 40}


def canonical(rel):
    """rel with its parent folders resolved, so that nothing is stored through a linked folder (tar refuses to extract
    it). The last component stays as it is, since it may be a link that has to be kept."""
    parent = os.path.realpath(os.path.join(TM, os.path.dirname(rel)))
    return os.path.normpath(os.path.join(os.path.relpath(parent, os.path.realpath(TM)), os.path.basename(rel)))


def walk(rel, files, links):
    """Add rel (file, dir or link, relative to trained_models/) and everything it reaches."""
    rel = canonical(rel)
    p = os.path.join(TM, rel)
    if rel in files or rel in links:
        return
    if os.path.islink(p):
        tgt = os.path.normpath(os.path.join(os.path.dirname(rel), os.readlink(p)))
        assert not tgt.startswith('..'), f'link outside trained_models/: {rel} -> {os.readlink(p)}'
        tgt = canonical(tgt)
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def single_driver_targets():
    """Targets of the (git-tracked) links in results/.../selected, relative to trained_models/."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ONE_SEL):
        for n in dirnames + filenames:
            p = os.path.join(dirpath, n)
            if os.path.islink(p):
                tgt = os.path.normpath(os.path.join(dirpath, os.readlink(p)))
                out.append(os.path.relpath(tgt, TM))
            elif os.path.isfile(p) and n != '.DS_Store':
                sys.exit(f'unexpected regular file, it would not reach the archive: {p}')
    return sorted(out)


def build_manifest(one_targets):
    """One row per agent. Checks that each copy is the seed it declares and each single-driver link the chosen seed."""
    rows, problems = [], []
    for scope, tree in MULTI.items():
        for f in sorted(glob.glob(os.path.join(TM, tree, 'selected', '*', '*', 'SELECTED_SEED.txt'))):
            race, algo = f.split(os.sep)[-3:-1]
            seed = int(re.match(r'seed\s+(\d+)', open(f).read()).group(1))
            model = os.path.join(os.path.dirname(f), 'best_model.zip')
            digest = sha256(model)
            source = os.path.join(TM, tree, f'seed{seed:02d}', race, algo, 'best_model.zip')
            if not os.path.isfile(source) or sha256(source) != digest:
                problems.append(f'{os.path.relpath(model, TM)} is not seed {seed}')
            rows.append([scope, race, algo, seed, os.path.relpath(model, ROOT), digest])
    chosen = {(r['gp'].replace(' Grand Prix', ''), r['algo']): int(r['best_seed'])
              for r in csv.DictReader(open(os.path.join(ONE_DIR, 'selection_20261028.tsv')), delimiter='\t')}
    for rel in one_targets:
        if not rel.endswith('best_model.zip'):
            continue
        parts = rel.split(os.sep)   # <tree>/seedNN/one_driver/<GP>/<ALGO>/best_model.zip
        seed, race, algo = int(parts[1][4:]), parts[3], parts[4]
        if chosen.get((race, algo)) != seed:
            problems.append(f'{rel}: selection_20261028.tsv says seed {chosen.get((race, algo))}')
        rows.append(['single_driver', race, algo, seed, os.path.join('trained_models', rel),
                     sha256(os.path.join(TM, rel))])
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default='f1-pit-stop-drl-v2.0.0-models.tar.gz')
    a = ap.parse_args()
    files, links = {}, {}
    for t in TREES:
        walk(t, files, links)
    one_targets = single_driver_targets()
    for rel in one_targets:
        walk(rel, files, links)

    def resolved(rel):
        return resolved(links[rel]) if rel in links else rel
    bad = [l for l in links if resolved(l) not in files and not any(f.startswith(resolved(l) + '/') for f in files)]
    rows, problems = build_manifest(one_targets)
    problems += [f'{e} is stored through a linked folder' for e in set(files) | set(links) if canonical(e) != e]
    for scope, n in EXPECTED_ZIPS.items():
        got = sum(r[0] == scope for r in rows)
        if got != n:
            problems.append(f'{scope}: {got} agents, expected {n}')
    groups = {}
    for f, size in files.items():
        groups[f.split('/')[0]] = groups.get(f.split('/')[0], 0) + size
    print(f'{len(files)} files, {sum(files.values()) / 1e6:.1f} MB, {len(links)} links')
    for g, size in sorted(groups.items()):
        print(f'  {g}: {size / 1e6:.1f} MB')
    print('agents:', {s: sum(r[0] == s for r in rows) for s in EXPECTED_ZIPS})
    print('unresolved links:', bad or 'none')
    print('problems:', problems or 'none')
    if bad or problems:
        sys.exit(1)
    if a.dry_run:
        return
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['scope', 'race', 'algorithm', 'seed', 'model_path', 'sha256'])
    w.writerows(rows)
    data = buf.getvalue().encode()
    with tarfile.open(a.out, 'w:gz') as tar:
        for rel in sorted(set(files) | set(links)):
            tar.add(os.path.join(TM, rel), arcname=os.path.join('trained_models', rel), recursive=False)
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print('->', a.out, f'{os.path.getsize(a.out) / 1e6:.1f} MB', 'sha256', sha256(a.out))


if __name__ == '__main__':
    main()
