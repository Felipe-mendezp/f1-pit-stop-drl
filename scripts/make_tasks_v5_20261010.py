"""2026-10-10 — Tareas del reentrenamiento con coeficientes de piloto corregidos (etapa C del camino 2).

Mismas 160 tareas que reproduce/cluster/tasks_v4_opcion3.tsv (scripts/make_tasks_v4_20261007.py:
8 GP x {PPO, RecurrentPPO} x semillas 1-10, mismos scripts, hiperparametros y fisica), cambiando solo:
    F1_SIMULATION_DIR = trained_models/simulation_v5_20261010
                        (scripts/build_simulation_v5_20261010.py)
    salida            = trained_models/rl_agents_v5_20261010/seed<ss>/
Escribe reproduce/cluster/tasks_v5.tsv (lo lee reproduce/cluster/run_task_v3_20261007.sh).

    python scripts/make_tasks_v5_20261010.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'reproduce', 'cluster', 'tasks_v4_opcion3.tsv')
DST = os.path.join(ROOT, 'reproduce', 'cluster', 'tasks_v5.tsv')
OLD_SIM, NEW_SIM = 'trained_models/simulation_yfv4_opcion3_20261007', 'trained_models/simulation_v5_20261010'
OLD_OUT, NEW_OUT = 'trained_models/rl_agents_v4_opcion3_20261007', 'trained_models/rl_agents_v5_20261010'


def main():
    assert os.path.isdir(os.path.join(ROOT, NEW_SIM)), NEW_SIM
    lines = []
    for line in open(SRC).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert sim == OLD_SIM and log.startswith(OLD_OUT + '/') and cmd.count(OLD_OUT) == 1, line
        lines.append('\t'.join(['v5_' + name, log.replace(OLD_OUT, NEW_OUT).replace('/opcion3__', '/v5_opcion3__'),
                                NEW_SIM, cmd.replace(OLD_OUT, NEW_OUT)]))
    assert len(lines) == 160 and not any(OLD_OUT in l or OLD_SIM + '\t' in l for l in lines)
    with open(DST, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{DST}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
