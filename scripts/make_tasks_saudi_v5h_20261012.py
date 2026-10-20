"""2026-10-12 — Tareas de Saudi con la "opcion 3 jerarquica" de neumaticos (v5h): paso 4 del estudio de Saudi.

Mismos comandos (scripts, hiperparametros, fisica, semillas) que las tareas v5 de Saudi, cambiando solo el
simulador y la carpeta de salida (scripts/build_simulation_v5h_20261011.py construyo los arboles):

  multi-piloto (Figs. 6/7): filas de Saudi de tasks_v5.tsv (PPO, RecurrentPPO) y tasks_v5_others.tsv
      (DQN, A2C, TRPO), semillas 1-10 -> 50 tareas
      F1_SIMULATION_DIR = trained_models/simulation_v5h_20261011
      salida            = trained_models/rl_agents_v5h_20261011/seed<ss>/Saudi_Arabian_NOR/<algo>/
      -> reproduce/cluster/tasks_v5h_saudi.tsv   (sbatch train_v3_20261007.sbatch)
  mono-piloto (Fig. 5 / Tabla N.1): filas de Saudi de tasks_onedriver_v5.tsv, semillas 1-5 -> 25 tareas
      F1_SIMULATION_DIR = trained_models/simulation_onedriver_opcion3h_20261011
      salida            = trained_models/rl_agents_onedriver_v5h_20261011/seed<ss>/one_driver/Saudi_Arabian/<algo>/
      -> reproduce/cluster/tasks_onedriver_v5h_saudi.tsv   (sbatch train_onedriver_20261011.sbatch)

    python scripts/make_tasks_saudi_v5h_20261012.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
JOBS = [  # (fuentes, sim viejo, sim nuevo, salida vieja, salida nueva, prefijo viejo, prefijo nuevo, destino, n)
    (['tasks_v5.tsv', 'tasks_v5_others.tsv'],
     'trained_models/simulation_v5_20261010', 'trained_models/simulation_v5h_20261011',
     'trained_models/rl_agents_v5_20261010', 'trained_models/rl_agents_v5h_20261011',
     'v5_opcion3__', 'v5h__', 'tasks_v5h_saudi.tsv', 50),
    (['tasks_onedriver_v5.tsv'],
     'trained_models/simulation_onedriver_opcion3_20261011', 'trained_models/simulation_onedriver_opcion3h_20261011',
     'trained_models/rl_agents_onedriver_v5_20261011', 'trained_models/rl_agents_onedriver_v5h_20261011',
     'onedriver_v5__', 'onedriver_v5h__', 'tasks_onedriver_v5h_saudi.tsv', 25),
]


def main():
    for srcs, s0, s1, o0, o1, p0, p1, dst, n in JOBS:
        assert os.path.isdir(os.path.join(ROOT, s1)), s1
        lines = []
        for src in srcs:
            for line in open(os.path.join(CL, src)).read().splitlines():
                name, log, sim, cmd = line.split('\t')
                if "'Saudi Arabian Grand Prix'" not in cmd:
                    continue
                assert sim == s0 and name.startswith(p0) and o0 in log and o0 in cmd, line
                lines.append('\t'.join([p1 + name[len(p0):], log.replace(o0, o1).replace(p0, p1), s1,
                                        cmd.replace(o0, o1)]))
        assert len(lines) == n and not any(o0 + '/' in l or s0 + '\t' in l for l in lines), len(lines)
        with open(os.path.join(CL, dst), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{dst}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
