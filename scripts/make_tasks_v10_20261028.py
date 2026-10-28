"""2026-10-28 — Tareas del multi-piloto v10 definitivo (Figs. 6-7): las 400 de reproduce/cluster/tasks_v10.tsv (27-oct, mismo
orden de indices) con el arbol trained_models/simulation_v10_20261028 (pilotos de la Ec. 2 v10) y salida en
trained_models/retrained_v10_20261028/. El arreglo del 27-oct se cancelo (pilotos de v6).
    python scripts/make_tasks_v10_20261028.py   -> reproduce/cluster/tasks_v10_20261028.tsv
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
OLD, NEW = 'retrained_v10_20261027', 'retrained_v10_20261028'
SIM_OLD, SIM_NEW = 'trained_models/simulation_v10_20261027', 'trained_models/simulation_v10_20261028'
dst = os.path.join(CL, 'tasks_v10_20261028.tsv')
assert not os.path.exists(dst), dst
out = []
for line in open(os.path.join(CL, 'tasks_v10.tsv')).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    assert sim == SIM_OLD and OLD in cmd and OLD in log, line
    out.append('\t'.join([name, log.replace(OLD, NEW), SIM_NEW, cmd.replace(OLD, NEW)]))
assert len(out) == 400 and not any(OLD in l or SIM_OLD in l for l in out)
open(dst, 'w').write('\n'.join(out) + '\n')
print(len(out), 'tareas ->', dst)
