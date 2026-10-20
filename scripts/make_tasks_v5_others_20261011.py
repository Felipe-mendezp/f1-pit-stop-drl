"""2026-10-11 — Tareas v5 para DQN, A2C y TRPO (los otros 3 algoritmos del paper).

Mismo protocolo que scripts/make_tasks_v4_20261007.py (filas multi_driver de reproduce/MODELS_MANIFEST.csv,
8 GP sin Emilia Romagna, mismo script e hiperparametros, semillas 1-10, fisica env_paper_v3, sin
--checkpoint-freq), con el simulador v5 (opcion3 + coeficientes de piloto Ec. 2):
    F1_SIMULATION_DIR = trained_models/simulation_v5_20261010
    salida            = trained_models/rl_agents_v5_20261010/seed<ss>/<GP_piloto>/<algo>/
(el mismo arbol que PPO/RPPO v5; cada algoritmo escribe en su propia carpeta).
Escribe reproduce/cluster/tasks_v5_others.tsv (240 tareas; la lee reproduce/cluster/run_task_v3_20261007.sh).

    python scripts/make_tasks_v5_others_20261011.py [--algos DQN A2C TRPO] [--seeds 1 ... 10]
        [--timesteps N --tag _smoke --out-base <dir>]
"""
import argparse, csv, os, re, shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'
SIM = 'trained_models/simulation_v5_20261010'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--algos', nargs='+', default=['DQN', 'A2C', 'TRPO'])
    ap.add_argument('--timesteps', type=int, default=None, help='solo para pruebas cortas')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out-base', default='trained_models/rl_agents_v5_20261010')
    a = ap.parse_args()
    assert os.path.isdir(os.path.join(ROOT, SIM)), SIM
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST.csv')))
            if r['scope'] == 'multi_driver' and r['algorithm'] in a.algos
            and 'Emilia_Romagna' not in r['model_path']]
    assert len(rows) == 8 * len(a.algos), len(rows)
    lines = []
    for s in a.seeds:
        out = f'{a.out_base}/seed{s:02d}'
        for r in rows:
            args = r['args'].replace('{PHYS}', PHYS).replace('{OUT}', shlex.quote(out))
            args = re.sub(r'--seeds? \d+', lambda m: m.group(0).split()[0] + f' {s}', args)
            args = re.sub(r'\s*--checkpoint-freq \d+', '', args)
            if a.timesteps:
                args += f' --timesteps {a.timesteps}'
            assert f' {s} ' in args + ' ', args
            cmd = ' '.join(shlex.quote(c) for c in ['scripts/' + os.path.basename(r['script'])] + shlex.split(args))
            name = f"v5_opcion3__seed{s:02d}__{r['model_path'].split('/', 1)[1].replace('/', '__')}"
            lines.append('\t'.join([name, f'{a.out_base}/logs/{name}.log', SIM, cmd]))
    path = os.path.join(ROOT, 'reproduce', 'cluster', f'tasks_v5_others{a.tag}.tsv')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{path}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
