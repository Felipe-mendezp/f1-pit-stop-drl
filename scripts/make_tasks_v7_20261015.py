"""2026-10-15 — Tareas de entrenamiento v7 (Fase 3 de docs/DISENO_TYRELIFE_2021_20261015.md).

Multi-piloto: las mismas 400 tareas de reproduce/cluster/tasks_v6.tsv (8 GP, pilotos v5hd, abandonos de rivales via
scripts/rival_dnf_v6_20261014.py, 5 algoritmos x 10 semillas, mismos hiperparametros y semillas), con el simulador
trained_models/simulation_v7_20261015 y salida trained_models/rl_agents_v7_20261015/.
Mono-piloto: 8 GP x 5 algoritmos x semillas 1-5 con scripts/one_driver_v7_20261015.py train (sin filtro, mismos
argumentos que tasks_onedriver_v5hnf.tsv), simulador trained_models/simulation_onedriver_v7_20261015, salida
trained_models/rl_agents_onedriver_v7_20261015/.

    python scripts/make_tasks_v7_20261015.py  ->  reproduce/cluster/tasks_v7.tsv, tasks_onedriver_v7.tsv
"""
import os, shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
SIM6, SIM7 = 'trained_models/simulation_v6_20261014', 'trained_models/simulation_v7_20261015'
O6, O7 = 'trained_models/rl_agents_v6_20261014', 'trained_models/rl_agents_v7_20261015'
SIM1 = 'trained_models/simulation_onedriver_v7_20261015'
O1 = 'trained_models/rl_agents_onedriver_v7_20261015'
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
ALGOS = ['RecurrentPPO', 'PPO', 'DQN', 'A2C', 'TRPO']


def multi():
    lines = []
    for line in open(os.path.join(CL, 'tasks_v6.tsv')).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert name.startswith('v6__') and sim == SIM6 and O6 in log and O6 in cmd, line
        name = 'v7__' + name[len('v6__'):]
        log = log.replace(O6, O7).replace('/v6__', '/v7__')
        cmd = cmd.replace(O6, O7)
        lines.append('\t'.join([name, log, SIM7, cmd]))
    assert len(lines) == 400 and len({l.split('\t')[0] for l in lines}) == 400
    assert not any('v6_20261014' in l.replace('rival_dnf_v6_20261014', '') for l in lines)
    return lines


def one():
    lines = []
    for s in [1, 2, 3, 4, 5]:
        for gp in GPS:
            for algo in ALGOS:
                name = f"onedriver_v7__seed{s:02d}__{gp[:-11].replace(' ', '_')}__{algo}"
                cmd = ['scripts/one_driver_v7_20261015.py', 'train', '--agents-dir', f'{O1}/seed{s:02d}',
                       '--gps', gp, '--algos', algo, '--seed', str(s), '--timesteps', '5000000',
                       '--n-envs', '10', '--torch-threads', '2', '--vec', 'dummy']
                lines.append('\t'.join([name, f'{O1}/logs/{name}.log', SIM1, ' '.join(shlex.quote(c) for c in cmd)]))
    return lines


def main():
    for fname, lines in [('tasks_v7.tsv', multi()), ('tasks_onedriver_v7.tsv', one())]:
        with open(os.path.join(CL, fname), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{fname}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
