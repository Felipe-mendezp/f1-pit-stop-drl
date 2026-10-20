"""2026-10-12 — Tareas (multi- y mono-piloto) de los 5 algoritmos en los 7 GP distintos de Saudi con la opcion 3 jerarquica en los
8 GP (simulador scripts/build_simulation_v5h8_20261012.py). Mismos comandos que las filas de
tasks_v5.tsv, cambiando solo el simulador y la carpeta de salida; la salida comparte arbol con las
tareas de Saudi (make_tasks_saudi_v5h_20261012.py), cuyo GP es identico en v5h8:
    F1_SIMULATION_DIR = trained_models/simulation_v5h8_20261012
    salida            = trained_models/rl_agents_v5h_20261011/seed<ss>/<GP_piloto>/<algo>/
    tasks_v5.tsv        -> reproduce/cluster/tasks_v5h8_ppo.tsv     (PPO, RecurrentPPO: 140 tareas)
    tasks_v5_others.tsv -> reproduce/cluster/tasks_v5h8_others.tsv  (DQN, A2C, TRPO: 210 tareas)
    (ambas con sbatch train_v3_20261007.sbatch)
Mono-piloto (Fig. 5 / Tabla N.1), filas no-Saudi de tasks_onedriver_v5.tsv (5 algoritmos x 5 semillas):
    F1_SIMULATION_DIR = trained_models/simulation_onedriver_opcion3h8_20261012
    salida            = trained_models/rl_agents_onedriver_v5h_20261011/seed<ss>/one_driver/<GP>/<algo>/
                        (mismo arbol que Saudi, make_tasks_saudi_v5h_20261012.py)
    -> reproduce/cluster/tasks_onedriver_v5h8.tsv  (175 tareas; sbatch train_onedriver_20261011.sbatch)

    python scripts/make_tasks_v5h8_20261012.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
MULTI = ('trained_models/simulation_v5_20261010', 'trained_models/simulation_v5h8_20261012',
         'trained_models/rl_agents_v5_20261010', 'trained_models/rl_agents_v5h_20261011', 'v5_opcion3__', 'v5h__')
ONE = ('trained_models/simulation_onedriver_opcion3_20261011', 'trained_models/simulation_onedriver_opcion3h8_20261012',
       'trained_models/rl_agents_onedriver_v5_20261011', 'trained_models/rl_agents_onedriver_v5h_20261011',
       'onedriver_v5__', 'onedriver_v5h__')
JOBS = [('tasks_v5.tsv', 'tasks_v5h8_ppo.tsv', 140, MULTI), ('tasks_v5_others.tsv', 'tasks_v5h8_others.tsv', 210, MULTI),
        ('tasks_onedriver_v5.tsv', 'tasks_onedriver_v5h8.tsv', 175, ONE)]


def main():
    for src, dst, n, (S0, S1, O0, O1, P0, P1) in JOBS:
        assert os.path.isdir(os.path.join(ROOT, S1)), S1
        lines = []
        for line in open(os.path.join(CL, src)).read().splitlines():
            name, log, sim, cmd = line.split('\t')
            if "'Saudi Arabian Grand Prix'" in cmd:
                continue
            assert sim == S0 and name.startswith(P0) and O0 in log and O0 in cmd, line
            lines.append('\t'.join([P1 + name[len(P0):], log.replace(O0, O1).replace(P0, P1), S1, cmd.replace(O0, O1)]))
        assert len(lines) == n and not any(O0 + '/' in l or S0 + '\t' in l for l in lines), len(lines)
        with open(os.path.join(CL, dst), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{dst}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
