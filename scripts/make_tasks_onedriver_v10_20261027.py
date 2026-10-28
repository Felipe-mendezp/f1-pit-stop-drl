"""2026-10-27 — Tareas del mono-piloto v10 (Fig. 5 / Tabla N.1): las mismas 200 de reproduce/cluster/tasks_onedriver_v9r2.tsv
(8 GP x 5 algoritmos x semillas 1-5, mismos argumentos; envoltorio scripts/one_driver_v10_20261027.py) con el arbol simulation_onedriver_v10_20261027 y
salida en trained_models/retrained_v10_onedriver_20261027/.
    python scripts/make_tasks_onedriver_v10_20261027.py   -> reproduce/cluster/tasks_onedriver_v10.tsv
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(ROOT, 'reproduce', 'cluster', 'tasks_onedriver_v9r2.tsv')
dst = os.path.join(ROOT, 'reproduce', 'cluster', 'tasks_onedriver_v10.tsv')
assert not os.path.exists(dst), dst
out = []
for line in open(src).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    assert sim == 'trained_models/simulation_onedriver_v7_20261015', sim
    name = name.replace('onedriver_v9r2__', 'onedriver_v10__')
    log = log.replace('rl_agents_onedriver_v9r2_20261022', 'retrained_v10_onedriver_20261027').replace('onedriver_v9r2__', 'onedriver_v10__')
    cmd = cmd.replace('rl_agents_onedriver_v9r2_20261022', 'retrained_v10_onedriver_20261027').replace('scripts/one_driver_v8_20261019.py', 'scripts/one_driver_v10_20261027.py')
    assert 'v9r2_2026' not in log and 'rl_agents_onedriver' not in cmd
    out.append('\t'.join([name, log, 'trained_models/simulation_onedriver_v10_20261027', cmd]))
open(dst, 'w').write('\n'.join(out) + '\n')
print(len(out), 'tareas ->', dst)
