"""2026-10-28 — Tareas del simulador v10eGQ (edad efectiva GQ, adoptada por el autor; arboles de
scripts/build_simulation_v10e_20261028.py GQ; edades de los juegos 2024 con scripts/tyre_ages_v10eGQ_20261028.py, que se
antepone a cada comando). Copia de make_tasks_v10d_20261028.py a partir de las tareas v10d (mismo orden y semillas):
  multi-piloto: reproduce/cluster/tasks_v10d_20261028.tsv -> reproduce/cluster/tasks_v10eGQ_20261028.tsv
                (arbol simulation_v10eGQ_20261028, salida trained_models/retrained_v10eGQ_20261028/)
  mono-piloto:  reproduce/cluster/tasks_onedriver_v10d_20261028.tsv -> reproduce/cluster/tasks_onedriver_v10eGQ_20261028.tsv
                (arbol simulation_onedriver_v10eGQ_20261028, envoltorio scripts/one_driver_v10eGQ_20261028.py, salida
                trained_models/retrained_v10eGQ_onedriver_20261028/)
    python scripts/make_tasks_v10eGQ_20261028.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
W = 'scripts/tyre_ages_v10eGQ_20261028.py '
JOBS = [('tasks_v10d_20261028.tsv', 'tasks_v10eGQ_20261028.tsv', 400, 'scripts/v10_wrapper_20261027.py ',
         [('trained_models/simulation_v10d_20261028', 'trained_models/simulation_v10eGQ_20261028'),
          ('retrained_v10d_20261028', 'retrained_v10eGQ_20261028'), ('v10d__', 'v10eGQ__')]),
        ('tasks_onedriver_v10d_20261028.tsv', 'tasks_onedriver_v10eGQ_20261028.tsv', 200, 'scripts/tyre_reuse_v9r2_20261022.py ',
         [('trained_models/simulation_onedriver_v10d_20261028', 'trained_models/simulation_onedriver_v10eGQ_20261028'),
          ('retrained_v10d_onedriver_20261028', 'retrained_v10eGQ_onedriver_20261028'),
          ('scripts/one_driver_v10d_20261028.py', 'scripts/one_driver_v10eGQ_20261028.py'), ('onedriver_v10d__', 'onedriver_v10eGQ__')])]
for src, dst, n, first, rep in JOBS:
    dst = os.path.join(CL, dst)
    assert not os.path.exists(dst), dst
    out = []
    for line in open(os.path.join(CL, src)).read().splitlines():
        for a, b in rep:
            assert a in line, (a, line)
            line = line.replace(a, b)
        f = line.split('\t')
        assert len(f) == 4 and f[3].startswith(first), f
        f[3] = W + f[3]
        out.append('\t'.join(f))
    assert len(out) == n and not any(a in l for l in out for a, _ in rep)
    open(dst, 'w').write('\n'.join(out) + '\n')
    print(len(out), 'tareas ->', dst)
