"""Seleccion de semillas mono-piloto v5h SIN filtro (2026-10-14).

Igual que results/onedriver_v5h_20261013/select_seeds_20261013.py (mejor semilla sane = menor gap vs QMIP; empate ->
semilla menor), combinando dos fuentes:
  * Saudi, Miami, Hungarian, Dutch y US (los 5 GP que tenian filtro): screen_seed0X.tsv de esta carpeta y agentes
    trained_models/rl_agents_onedriver_v5hnf_20261014;
  * Bahrain, Belgian y Singapore: el filtro 5 % no quitaba ningun compuesto ahi, asi que se reutilizan los
    screenings y agentes v5h (results/onedriver_v5h_20261013, trained_models/rl_agents_onedriver_v5h_20261011).
Escribe selection_20261014.tsv y selected/one_driver/<GP corto>/<algo>/best_model.zip (symlinks).
"""
import os, csv, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = {'nf': (HERE, os.path.join(ROOT, 'trained_models', 'rl_agents_onedriver_v5hnf_20261014')),
       'v5h': (os.path.join(ROOT, 'results', 'onedriver_v5h_20261013'),
               os.path.join(ROOT, 'trained_models', 'rl_agents_onedriver_v5h_20261011'))}
REUSE = {'Bahrain Grand Prix', 'Belgian Grand Prix', 'Singapore Grand Prix'}
SEEDS = ['01', '02', '03', '04', '05']

rows = {}
for src, (sdir, _) in SRC.items():
    for s in SEEDS:
        with open(os.path.join(sdir, f'screen_seed{s}.tsv')) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if (r['gp'] in REUSE) == (src == 'v5h'):
                    rows.setdefault((r['gp'], r['algo']), []).append((s, r, src))

out = []
for (gp, algo), lst in rows.items():
    assert len(lst) == 5 and all(r['flag'] != 'MISSING' for _, r, _ in lst), (gp, algo)
    src = lst[0][2]
    gaps = [float(r['gap_s']) for _, r, _ in lst if r['gap_s'] != 'nan']
    sane = [(float(r['gap_s']), s) for s, r, _ in lst if r['flag'] == 'sane']
    best = min(sane) if sane else None
    out.append(dict(gp=gp, algo=algo, source=src, n_seeds=len(lst), n_sane=len(sane),
                    best_seed=best[1] if best else 'NONE', best_gap=f'{best[0]:.3f}' if best else 'nan',
                    median_all=f'{statistics.median(gaps):.3f}' if gaps else 'nan',
                    median_sane=f'{statistics.median([g for g, _ in sane]):.3f}' if sane else 'nan',
                    gaps=' '.join(f"{s}:{r['gap_s']}{'' if r['flag']=='sane' else '*'}" for s, r, _ in lst)))
    if best:
        short = gp[:-11]
        d = os.path.join(HERE, 'selected', 'one_driver', short, algo)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, 'best_model.zip')
        if os.path.lexists(link):
            os.remove(link)
        target = os.path.join(SRC[src][1], f'seed{best[1]}', 'one_driver', short, algo, 'best_model.zip')
        assert os.path.isfile(target), target
        os.symlink(os.path.relpath(target, d), link)

with open(os.path.join(HERE, 'selection_20261014.tsv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t'); w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['gp'][:-11]:14s} {r['algo']:13s} {r['source']:3s} sane {r['n_sane']}/{r['n_seeds']}  best s{r['best_seed']} "
          f"{r['best_gap']:>8s}  med {r['median_all']:>8s} (sane {r['median_sane']:>8s})  [{r['gaps']}]")
print('* = no sane')
