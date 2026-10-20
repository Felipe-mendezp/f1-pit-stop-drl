"""2026-10-19 — Accion de la vuelta 0 en el screening mono-piloto v8 (lap0_seedXX.tsv + screen_seedXX.tsv).

Por GP, lap0_seedXX.tsv tiene primero el episodio de la QMIP y luego uno por modelo existente, en el orden de las filas
de screen_seedXX.tsv (sin las MISSING). Se verifica que el compuesto anotado coincida con el 'start C?' del plan.
Escribe lap0_summary_20261019.tsv (gp, algo, semilla, accion, compuesto) e imprime los modelos con accion 0.
    python results/onedriver_v8_20261019/summarize_lap0_20261019.py
"""
import csv, os
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
rows, bad = [], 0
for s in ['01', '02', '03', '04', '05']:
    lap0 = [l.split('\t') for l in open(os.path.join(HERE, f'lap0_seed{s}.tsv')).read().splitlines()]
    scr = list(csv.DictReader(open(os.path.join(HERE, f'screen_seed{s}.tsv')), delimiter='\t'))
    i = 0
    for gp in dict.fromkeys(r['gp'] for r in scr):
        assert lap0[i][0] == gp, (s, gp, lap0[i]); i += 1          # QMIP
        for r in [r for r in scr if r['gp'] == gp and r['flag'] != 'MISSING']:
            g, act, comp = lap0[i]; i += 1
            assert g == gp
            start = r['plan'].split()[1]
            bad += start != comp
            rows.append(dict(gp=gp, algo=r['algo'], seed=s, action=act, compound=comp, flag=r['flag'], gap_s=r['gap_s']))
    assert i == len(lap0), (s, i, len(lap0))
with open(os.path.join(HERE, 'lap0_summary_20261019.tsv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} modelos; compuesto distinto del plan: {bad}')
print('acciones en la vuelta 0:', dict(Counter(r['action'] for r in rows)))
for r in rows:
    if r['action'] == '0':
        print(f"  accion 0: {r['gp'][:-11]:14s} {r['algo']:13s} seed{r['seed']}  {r['compound']}  {r['flag']}  gap {r['gap_s']}")
