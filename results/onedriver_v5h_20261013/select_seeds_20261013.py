"""Seleccion de semillas mono-piloto v5h (2026-10-13).

Lee screen_seed0{1..5}.tsv (scripts/one_driver_v5h_20261013.py screen) y, por (GP, algo):
  best   = semilla sane con menor gap (empate -> semilla menor); va a Fig. 5 / Tabla N.1
  median = mediana del gap sobre las 5 semillas (todas) y sobre las sane
Escribe selection_20261013.tsv y arma selected/one_driver/<GP corto>/<algo>/best_model.zip (symlinks).
"""
import os, sys, csv, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AG = os.path.join(ROOT, 'trained_models', 'rl_agents_onedriver_v5h_20261011')
SEEDS = ['01', '02', '03', '04', '05']

rows = {}
for s in SEEDS:
    with open(os.path.join(HERE, f'screen_seed{s}.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.setdefault((r['gp'], r['algo']), []).append((s, r))

out = []
for (gp, algo), lst in rows.items():
    gaps = [float(r['gap_s']) for _, r in lst if r['gap_s'] != 'nan']
    sane = [(float(r['gap_s']), s) for s, r in lst if r['flag'] == 'sane']
    best = min(sane) if sane else None
    out.append(dict(gp=gp, algo=algo, n_seeds=len(lst), n_sane=len(sane),
                    best_seed=best[1] if best else 'NONE', best_gap=f'{best[0]:.3f}' if best else 'nan',
                    median_all=f'{statistics.median(gaps):.3f}' if gaps else 'nan',
                    median_sane=f'{statistics.median([g for g, _ in sane]):.3f}' if sane else 'nan',
                    gaps=' '.join(f"{s}:{r['gap_s']}{'' if r['flag']=='sane' else '*'}" for s, r in lst)))
    if best:
        short = gp[:-11]
        d = os.path.join(HERE, 'selected', 'one_driver', short, algo)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, 'best_model.zip')
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(os.path.join(AG, f'seed{best[1]}', 'one_driver', short, algo, 'best_model.zip'), d), link)

with open(os.path.join(HERE, 'selection_20261013.tsv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t'); w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['gp'][:-11]:14s} {r['algo']:13s} sane {r['n_sane']}/{r['n_seeds']}  best s{r['best_seed']} {r['best_gap']:>8s}  "
          f"med {r['median_all']:>8s} (sane {r['median_sane']:>8s})  [{r['gaps']}]")
print('* = no sane')
