"""2026-10-19 — Tareas de entrenamiento mono-piloto v8.

Las mismas 200 tareas de reproduce/cluster/tasks_onedriver_v7.tsv (8 GP x 5 algoritmos x semillas 1-5, mismos
argumentos y simulador trained_models/simulation_onedriver_v7_20261015), lanzadas con scripts/one_driver_v8_20261019.py
en vez de scripts/one_driver_v7_20261015.py y con salida trained_models/rl_agents_onedriver_v8_20261019/.

    python scripts/make_tasks_onedriver_v8_20261019.py  ->  reproduce/cluster/tasks_onedriver_v8.tsv
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
O7, O8 = 'trained_models/rl_agents_onedriver_v7_20261015', 'trained_models/rl_agents_onedriver_v8_20261019'
W7, W8 = 'scripts/one_driver_v7_20261015.py', 'scripts/one_driver_v8_20261019.py'


def main():
    lines = []
    for line in open(os.path.join(CL, 'tasks_onedriver_v7.tsv')).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert name.startswith('onedriver_v7__') and O7 in log and O7 in cmd and cmd.startswith(W7 + ' '), line
        name = 'onedriver_v8__' + name[len('onedriver_v7__'):]
        log = log.replace(O7, O8).replace('/onedriver_v7__', '/onedriver_v8__')
        cmd = W8 + cmd[len(W7):].replace(O7, O8)
        lines.append('\t'.join([name, log, sim, cmd]))
    assert len(lines) == 200 and len({l.split('\t')[0] for l in lines}) == 200
    assert not any(O7 in l or 'onedriver_v7__' in l for l in lines)
    with open(os.path.join(CL, 'tasks_onedriver_v8.tsv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'tasks_onedriver_v8.tsv: {len(lines)} tareas')


if __name__ == '__main__':
    main()
