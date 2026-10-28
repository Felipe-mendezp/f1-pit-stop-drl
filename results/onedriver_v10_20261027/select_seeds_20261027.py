"""Seleccion de semillas mono-piloto v10 (2026-10-27; copia de ../onedriver_v9r2_20261022/select_seeds_20261022.py con el arbol v10). Original: v9r2 (2026-10-22; copia de ../onedriver_v9r_20261021/select_seeds_20261021.py con el arbol v9r2).

Original: Seleccion de semillas mono-piloto v9r (2026-10-21).

Mismo criterio que results/onedriver_v8_20261019/select_seeds_20261019.py (mejor semilla sane = menor gap frente a la
QMIP con juegos; empate -> semilla menor), con screen_seed0X.tsv de esta carpeta y agentes
trained_models/retrained_v10_onedriver_20261027. Nuevo en v9r: con events_seed0X.tsv (screen_v9r_20261021.py) se
verifica que el compuesto de partida anotado = el del plan y se cuentan los pedidos de compuesto sin juegos (vuelta 0
y carrera, -1000 cada uno, no incluidos en la gap); una semilla con alguno no es sane ('NO-SET') y sus paradas se
recuentan sin esos pedidos. Escribe selection_20261027.tsv, events_summary_20261027.tsv y
selected/one_driver/<GP corto>/<algo>/best_model.zip (symlinks).
"""
import os, csv, statistics
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AGENTS = os.path.join(ROOT, 'trained_models', 'retrained_v10_onedriver_20261027')
SEEDS = ['01', '02', '03', '04', '05']

rows, ev_rows, bad_start = {}, [], 0
for s in SEEDS:
    scr = list(csv.DictReader(open(os.path.join(HERE, f'screen_seed{s}.tsv')), delimiter='\t'))
    ev = [l.split('\t') for l in open(os.path.join(HERE, f'events_seed{s}.tsv')).read().splitlines()]
    eps, cur = [], None  # un episodio por linea L0, con sus NS
    for e in ev:
        if e[0] == 'L0':
            cur = dict(gp=e[1], action=e[2], compound=e[3], nos0=int(e[4]), ns=0); eps.append(cur)
        else:
            assert cur is not None and e[1] == cur['gp']; cur['ns'] += 1
    i = 0
    for gp in dict.fromkeys(r['gp'] for r in scr):
        q = eps[i]; i += 1                                            # QMIP
        assert q['gp'] == gp and q['nos0'] == 0 and q['ns'] == 0, ('QMIP sin juego', s, gp)
        for r in [r for r in scr if r['gp'] == gp]:
            if r['flag'] != 'MISSING':
                e = eps[i]; i += 1
                assert e['gp'] == gp
                bad_start += r['plan'].split()[1] != e['compound']
                n_no = e['nos0'] + e['ns']
                if n_no:
                    r['stops'] = str(int(r['stops']) - e['ns'])
                    r['flag'] = 'NO-SET'
                ev_rows.append(dict(gp=gp, algo=r['algo'], seed=s, lap0_action=e['action'], start=e['compound'],
                                    no_set_lap0=e['nos0'], no_set_race=e['ns'], flag=r['flag'], gap_s=r['gap_s']))
            rows.setdefault((gp, r['algo']), []).append((s, r))
    assert i == len(eps), (s, i, len(eps))

out = []
for (gp, algo), lst in rows.items():
    miss = [s for s, r in lst if r['flag'] == 'MISSING']
    gaps = [float(r['gap_s']) for _, r in lst if r['gap_s'] != 'nan']
    sane = [(float(r['gap_s']), s) for s, r in lst if r['flag'] == 'sane']
    best = min(sane) if sane else None
    out.append(dict(gp=gp, algo=algo, n_seeds=len(lst), n_missing=len(miss), n_sane=len(sane),
                    best_seed=best[1] if best else 'NONE', best_gap=f'{best[0]:.3f}' if best else 'nan',
                    median_all=f'{statistics.median(gaps):.3f}' if gaps else 'nan',
                    median_sane=f'{statistics.median([g for g, _ in sane]):.3f}' if sane else 'nan',
                    gaps=' '.join(f"{s}:{r['gap_s']}{'' if r['flag'] == 'sane' else '*'}" for s, r in lst)))
    if best:
        short = gp[:-11]
        d = os.path.join(HERE, 'selected', 'one_driver', short, algo)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, 'best_model.zip')
        if os.path.lexists(link):
            os.remove(link)
        target = os.path.join(AGENTS, f'seed{best[1]}', 'one_driver', short, algo, 'best_model.zip')
        assert os.path.isfile(target), target
        os.symlink(os.path.relpath(target, d), link)

with open(os.path.join(HERE, 'selection_20261027.tsv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t'); w.writeheader(); w.writerows(out)
with open(os.path.join(HERE, 'events_summary_20261027.tsv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(ev_rows[0]), delimiter='\t'); w.writeheader(); w.writerows(ev_rows)
for r in out:
    print(f"{r['gp'][:-11]:14s} {r['algo']:13s} sane {r['n_sane']}/{r['n_seeds']} (faltan {r['n_missing']})  "
          f"best s{r['best_seed']} {r['best_gap']:>8s}  med {r['median_all']:>8s} (sane {r['median_sane']:>8s})  [{r['gaps']}]")
print('* = no sane')
print(f"{len(ev_rows)} modelos; compuesto de partida distinto del plan: {bad_start}; "
      f"con pedidos sin juegos: {sum(1 for e in ev_rows if e['no_set_lap0'] or e['no_set_race'])}; "
      f"accion en la vuelta 0: {dict(Counter(e['lap0_action'] for e in ev_rows))}")
