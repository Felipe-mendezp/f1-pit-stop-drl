"""2026-10-29 — Tareas del multi-piloto v10eGQ con el adelantamiento v10f (logit con etiqueta por par y vuelta L+1 sin
SC/VSC/roja, aplicado a todos los gaps sobre el gap medio, y gap de seguimiento 0.4 + Exp(0.26); ver
results/sim_v10eGQ_20261028/README_adelantamiento_20261029.md). A partir de reproduce/cluster/tasks_v10eGQ_20261028.tsv
(mismo orden, semillas y arbol simulation_v10eGQ_20261028, reconstruido el 29-oct con el logit nuevo): antepone
scripts/overtake_v10f_20261029.py, --following-gap-scale 1.0 -> 0.26 y salida trained_models/retrained_v10eGQf_20261029/
(carpeta nueva: la prueba 18033848 dejo salidas con la fisica anterior en retrained_v10eGQ_20261028/).
El mono-piloto no usa adelantamientos (su arbol queda identico): siguen valiendo tasks_onedriver_v10eGQ_20261028.tsv.
    python scripts/make_tasks_v10eGQf_20261029.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
SRC, DST = os.path.join(CL, 'tasks_v10eGQ_20261028.tsv'), os.path.join(CL, 'tasks_v10eGQf_20261029.tsv')
W = 'scripts/overtake_v10f_20261029.py '
REP = [('retrained_v10eGQ_20261028', 'retrained_v10eGQf_20261029'), ('v10eGQ__', 'v10eGQf__'),
       ('--following-gap-scale 1.0 ', '--following-gap-scale 0.26 ')]
assert not os.path.exists(DST), DST
out = []
for line in open(SRC).read().splitlines():
    for a, b in REP:
        assert a in line, (a, line)
        line = line.replace(a, b)
    f = line.split('\t')
    assert len(f) == 4 and f[2] == 'trained_models/simulation_v10eGQ_20261028' and f[3].startswith('scripts/tyre_ages_v10eGQ_20261028.py '), f
    f[3] = W + f[3]
    out.append('\t'.join(f))
assert len(out) == 400 and not any(a in l for l in out for a, _ in REP[:2])
open(DST, 'w').write('\n'.join(out) + '\n')
print(len(out), 'tareas ->', DST)
