"""2026-10-07 — [v4 = v3 con el modelo YF v4: simuladores simulation_yfv4_<calib>_20261007, ver scripts/fit_yf_model_v4_20261007.py] Lista de tareas del reentrenamiento multi-semilla v3 (SC revelado antes de decidir).

Para cada calibración, cada fila multi_driver de reproduce/MODELS_MANIFEST.csv (8 GP del paper, sin
Emilia Romagna) y cada semilla, replica la llamada del manifiesto (mismo script e hiperparámetros) con:
    semilla           = --seed / --seeds <s>  (reemplaza la semilla efectiva del manifiesto)
    física            = reproduce/env_paper_v3_20261007.sh (F1_SC_REVEAL=1, F1_LASTLAP_PITOUT=1, ...)
    F1_SIMULATION_DIR = trained_models/simulation_yfv4_<calib>_20261007
    salida            = trained_models/rl_agents_v4_<calib>_20261007/seed<ss>/
    sin --checkpoint-freq (los checkpoints intermedios ocupan ~93 MB por RPPO; best_model se guarda igual)

Escribe reproduce/cluster/tasks_v4_<calib>.tsv, una tarea por línea (la lee
reproduce/cluster/run_task_v3_20261007.sh con el índice del array de SLURM). Rutas relativas a la
raíz del repo, así el mismo archivo sirve en el Mac y en el cluster.

    python scripts/make_tasks_v4_20261007.py [--calibs control opcion2 opcion3] [--seeds 1 ... 10]
        [--algos PPO RecurrentPPO DQN A2C TRPO] [--timesteps N --tag _smoke]
"""
import argparse, csv, os, re, shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'
SIM = {'control': 'trained_models/simulation_yfv4_control_20261007',
       'opcion2': 'trained_models/simulation_yfv4_opcion2_20261007',
       'opcion3': 'trained_models/simulation_yfv4_opcion3_20261007'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibs', nargs='+', default=['control', 'opcion2', 'opcion3'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--algos', nargs='+', default=['PPO', 'RecurrentPPO', 'DQN', 'A2C', 'TRPO'])
    ap.add_argument('--timesteps', type=int, default=None, help='solo para pruebas cortas')
    ap.add_argument('--tag', default='', help='sufijo de la salida, p.ej. _smoke')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST.csv')))
            if r['scope'] == 'multi_driver' and r['algorithm'] in a.algos
            and 'Emilia_Romagna' not in r['model_path']]
    os.makedirs(os.path.join(ROOT, 'reproduce', 'cluster'), exist_ok=True)
    for calib in a.calibs:
        assert os.path.isdir(os.path.join(ROOT, SIM[calib])), SIM[calib]
        base = f'trained_models/rl_agents_v4_{calib}_20261007{a.tag}'
        lines = []
        for s in a.seeds:
            out = f'{base}/seed{s:02d}'
            for r in rows:
                args = r['args'].replace('{PHYS}', PHYS).replace('{OUT}', shlex.quote(out))
                args = re.sub(r'--seeds? \d+', lambda m: m.group(0).split()[0] + f' {s}', args)
                args = re.sub(r'\s*--checkpoint-freq \d+', '', args)
                if a.timesteps:
                    args += f' --timesteps {a.timesteps}'
                assert f' {s} ' in args + ' ', args
                cmd = ' '.join(shlex.quote(c) for c in ['scripts/' + os.path.basename(r['script'])]
                               + shlex.split(args))
                name = f"{calib}__seed{s:02d}__{r['model_path'].split('/', 1)[1].replace('/', '__')}"
                log = f'{base}/logs/{name}.log'
                lines.append('\t'.join([name, log, SIM[calib], cmd]))
        path = os.path.join(ROOT, 'reproduce', 'cluster', f'tasks_v4_{calib}{a.tag}.tsv')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{path}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
