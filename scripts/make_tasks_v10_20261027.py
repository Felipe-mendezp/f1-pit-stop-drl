"""2026-10-27 — Tareas del multi-piloto v10 (Figs. 6-7): las mismas 400 de reproduce/cluster/tasks_v9r2.tsv (GP, piloto,
algoritmos, semillas, hiperparametros) con el arbol trained_models/simulation_v10_20261027, el envoltorio
scripts/v10_wrapper_20261027.py antepuesto (VSC solo en la 1a vuelta; DummyVecEnv con F1_VEC_DUMMY=1 del sbatch) y salida
en trained_models/retrained_v10_20261027/.
    python scripts/make_tasks_v10_20261027.py   -> reproduce/cluster/tasks_v10.tsv
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
OLD, NEW = 'trained_models/rl_agents_v9r2_20261022', 'trained_models/retrained_v10_20261027'
dst = os.path.join(CL, 'tasks_v10.tsv')
assert not os.path.exists(dst), dst
out = []
for line in open(os.path.join(CL, 'tasks_v9r2.tsv')).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    assert name.startswith('v9r2__') and sim == 'trained_models/simulation_v7_20261015' and OLD in cmd, line
    assert cmd.startswith('scripts/tyre_reuse_v9r2_20261022.py '), cmd
    out.append('\t'.join(['v10__' + name[len('v9r2__'):], log.replace(OLD, NEW).replace('/v9r2__', '/v10__'),
                          'trained_models/simulation_v10_20261027', 'scripts/v10_wrapper_20261027.py ' + cmd.replace(OLD, NEW)]))
assert len(out) == 400 and not any(OLD in l or 'v9r2__' in l for l in out)
open(dst, 'w').write('\n'.join(out) + '\n')
print(len(out), 'tareas ->', dst)
