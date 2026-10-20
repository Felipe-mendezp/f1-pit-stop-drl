"""2026-10-14 — Tareas del mono-piloto v5h SIN filtro de compuestos (Fig. 5 / Tabla N.1): los 5 GP que tenian
filtro (Dutch, Hungarian, Miami, Saudi, US) x 5 algoritmos x semillas 1-5. Bahrain, Belgian y Singapore no cambian
(el filtro no excluia nada) y se reutilizan de trained_models/rl_agents_onedriver_v5h_20261011.

Cada tarea entrena un (GP, algoritmo, semilla) con scripts/one_driver_v5hnf_20261014.py train (hiperparametros
publicados, sin filtro, simulador trained_models/simulation_onedriver_opcion3h8_20261012).
La semilla del modelo es la que usa train_one_driver_paperhp (--seed s + indice del algoritmo).
Salida: trained_models/rl_agents_onedriver_v5hnf_20261014/seed<ss>/one_driver/<GP corto>/<algo>/best_model.zip
Escribe reproduce/cluster/tasks_onedriver_v5hnf.tsv (lo lee reproduce/cluster/run_task_v3_20261007.sh).

    python scripts/make_tasks_onedriver_v5hnf_20261014.py [--seeds 1 ... 5] [--timesteps N --tag _smoke]
"""
import argparse, os, shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = 'trained_models/simulation_onedriver_opcion3h8_20261012'
GPS = ['Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix', 'Dutch Grand Prix',
       'United States Grand Prix']
ALGOS = ['RecurrentPPO', 'PPO', 'DQN', 'A2C', 'TRPO']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--algos', nargs='+', default=ALGOS)
    ap.add_argument('--timesteps', type=int, default=5_000_000)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    assert os.path.isdir(os.path.join(ROOT, SIM)), SIM
    base = f'trained_models/rl_agents_onedriver_v5hnf_20261014{a.tag}'
    lines = []
    for s in a.seeds:
        for gp in GPS:
            for algo in a.algos:
                name = f"onedriver_v5hnf__seed{s:02d}__{gp[:-11].replace(' ', '_')}__{algo}"
                cmd = ['scripts/one_driver_v5hnf_20261014.py', 'train', '--agents-dir', f'{base}/seed{s:02d}',
                       '--gps', gp, '--algos', algo, '--seed', str(s), '--timesteps', str(a.timesteps),
                       '--n-envs', '10', '--torch-threads', '2', '--vec', 'dummy']
                lines.append('\t'.join([name, f'{base}/logs/{name}.log', SIM, ' '.join(shlex.quote(c) for c in cmd)]))
    path = os.path.join(ROOT, 'reproduce', 'cluster', f'tasks_onedriver_v5hnf{a.tag}.tsv')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{path}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
