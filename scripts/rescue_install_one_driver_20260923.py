"""Install rescued single-driver checkpoints — 2026-09-23.

Given the screening TSV of a rescue-seed tree, copy every SANE (GP, algo)
checkpoint directory into the target tree, moving the checkpoint it replaces to
<tree>/replaced_seed42/<GP>/<algo>/ (nothing is deleted), and append a line to
<tree>/RESCUE_NOTES.txt. Writes the still-unresolved pairs to --remaining-out.

Usage:
    python scripts/rescue_install_one_driver_20260923.py --tree <tree> \
        --seed-tree <seeds>/seed2024 --seed 2024 --screen <screen_seed2024.tsv> \
        --remaining-out <tsv>
"""
import argparse
import os
import shutil
import datetime as dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tree', required=True, help='target F1_RL_AGENTS_DIR (contains one_driver/)')
    ap.add_argument('--seed-tree', required=True, help='rescue F1_RL_AGENTS_DIR (contains one_driver/)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--screen', required=True, help='screening TSV of the rescue tree')
    ap.add_argument('--remaining-out', required=True)
    args = ap.parse_args()

    installed, remaining = [], []
    with open(args.screen) as f:
        header = f.readline()
        for line in f:
            gp, algo, gap, stops, changed, flag, plan = line.rstrip('\n').split('\t')
            gp_short = gp[:-11]
            src = os.path.join(args.seed_tree, 'one_driver', gp_short, algo)
            if flag != 'sane' or not os.path.exists(os.path.join(src, 'best_model.zip')):
                remaining.append((gp, algo, flag))
                continue
            dst = os.path.join(args.tree, 'one_driver', gp_short, algo)
            if os.path.exists(dst):
                bak = os.path.join(args.tree, 'replaced_seed42', gp_short, algo)
                os.makedirs(os.path.dirname(bak), exist_ok=True)
                if os.path.exists(bak):  # a previous rescue seed already moved it
                    bak = bak + f'_before_seed{args.seed}'
                shutil.move(dst, bak)
            shutil.copytree(src, dst)
            installed.append((gp, algo, gap, stops, plan))

    with open(os.path.join(args.tree, 'RESCUE_NOTES.txt'), 'a') as f:
        for gp, algo, gap, stops, plan in installed:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  {gp} / {algo}: installed seed {args.seed} "
                    f"(gap {gap} s, {stops} stops, {plan}); seed-42 checkpoint moved to replaced_seed42/\n")
    with open(args.remaining_out, 'w') as f:
        for gp, algo, flag in remaining:
            f.write(f"{gp}\t{algo}\n")
    print(f"installed {len(installed)}: {[(g, a) for g, a, *_ in installed]}")
    print(f"remaining {len(remaining)}: {remaining}")


if __name__ == '__main__':
    main()
