"""2026-10-28 — Tareas del simulador v10c (compuestos de FastF1 corregidos con OpenF1; scripts/build_simulation_v10c_20261028.py):
  multi-piloto: las 400 de reproduce/cluster/tasks_v10_20261028.tsv (mismo orden) con el arbol simulation_v10c_20261028 y
                salida trained_models/retrained_v10c_20261028/            -> reproduce/cluster/tasks_v10c_20261028.tsv
  mono-piloto:  las 200 de reproduce/cluster/tasks_onedriver_v10.tsv con el arbol simulation_onedriver_v10c_20261028, el
                envoltorio scripts/one_driver_v10c_20261028.py y salida trained_models/retrained_v10c_onedriver_20261028/
                                                                           -> reproduce/cluster/tasks_onedriver_v10c_20261028.tsv
    python scripts/make_tasks_v10c_20261028.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
JOBS = [('tasks_v10_20261028.tsv', 'tasks_v10c_20261028.tsv', 400,
         [('trained_models/simulation_v10_20261028', 'trained_models/simulation_v10c_20261028'),
          ('retrained_v10_20261028', 'retrained_v10c_20261028'), ('v10__', 'v10c__')]),
        ('tasks_onedriver_v10.tsv', 'tasks_onedriver_v10c_20261028.tsv', 200,
         [('trained_models/simulation_onedriver_v10_20261027', 'trained_models/simulation_onedriver_v10c_20261028'),
          ('retrained_v10_onedriver_20261027', 'retrained_v10c_onedriver_20261028'),
          ('scripts/one_driver_v10_20261027.py', 'scripts/one_driver_v10c_20261028.py'), ('onedriver_v10__', 'onedriver_v10c__')])]
for src, dst, n, rep in JOBS:
    dst = os.path.join(CL, dst)
    assert not os.path.exists(dst), dst
    out = []
    for line in open(os.path.join(CL, src)).read().splitlines():
        for a, b in rep:
            assert a in line, (a, line)
            line = line.replace(a, b)
        out.append(line)
    assert len(out) == n and not any(a in l for l in out for a, _ in rep)
    open(dst, 'w').write('\n'.join(out) + '\n')
    print(len(out), 'tareas ->', dst)
