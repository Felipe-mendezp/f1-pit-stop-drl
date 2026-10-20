"""Manifest de procedencia de los agentes multi-piloto v6 — 2026-10-14.

Una fila por agente elegido (trained_models/rl_agents_v6_20261014/selected/<GP_piloto>/<algo>), con las mismas columnas
que reproduce/MODELS_MANIFEST.csv (modelos publicados v2.0.0, que no se modifica). El comando de entrenamiento sale de
reproduce/cluster/tasks_v6.tsv (tarea de la semilla elegida) y la semilla de SELECTED_SEED.txt, que escribe
scripts/select_seeds_v5_20261010.py --calib v6 junto con el cumplimiento del filtro de validacion.

    python scripts/make_manifest_v6_20261014.py            # -> reproduce/MODELS_MANIFEST_v6_20261014.csv
    python scripts/make_manifest_v6_20261014.py --check    # solo verifica tasks_v6.tsv (sin modelos elegidos)
"""
import os, re, csv, shlex, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, 'reproduce', 'cluster', 'tasks_v6.tsv')
TREE = 'trained_models/rl_agents_v6_20261014'
OUT = os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST_v6_20261014.csv')
SIM = 'trained_models/simulation_v6_20261014'
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'
ENV = (f'source reproduce/env_paper_v3_20261007.sh; F1_SIMULATION_DIR={SIM}; '
       'F1_RIVAL_DNF=data/rival_dnf_2021_2023_20261014.json and v6 drivers (both set by the wrapper)')
COLS = ['scope', 'model_path', 'gp', 'driver', 'algorithm', 'effective_seed', 'script', 'args', 'env', 'notes']


def read_tasks():
    """name -> dict(gp_dir, algo, seed, script, args, gp, driver) desde tasks_v6.tsv."""
    out = {}
    with open(TASKS) as f:
        for line in f:
            if not line.strip():
                continue
            name, _log, sim, cmd = line.rstrip('\n').split('\t')
            assert sim == SIM, (name, sim)
            _, s, gp_dir, algo = name.split('__')
            tok = shlex.split(cmd)
            assert tok[0] == 'scripts/rival_dnf_v6_20261014.py', name
            script, args = tok[1], tok[2:]
            # semilla efectiva: --seed (train_rl_agents, un solo algoritmo => sin desfase) o --seeds (scripts *_fix)
            flag = '--seed' if '--seed' in args else '--seeds'
            seed = int(args[args.index(flag) + 1])
            assert seed == int(s[4:]), name
            gp = args[args.index('--gp' if '--gp' in args else '--gps') + 1]
            out[name] = dict(gp_dir=gp_dir, algo=algo, seed=seed, script=script, args=args, gp=gp,
                             driver='Driver_' + gp_dir.rsplit('_', 1)[1])
    return out


def template(args):
    """Argumentos con {PHYS} y {OUT} como en MODELS_MANIFEST.csv."""
    s = ' '.join(shlex.quote(a) if ' ' in a else a for a in args).replace("'", '"')
    s = s.replace(PHYS, '{PHYS}')
    s = re.sub(r'(--out-root|--output-dir) \S+', r'\1 {OUT}', s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    tasks = read_tasks()
    gps = sorted({t['gp_dir'] for t in tasks.values()})
    print(f'{len(tasks)} tareas, {len(gps)} GP: {gps}')
    if a.check:
        return
    rows, missing = [], []
    for gp_dir in gps:
        for algo in ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']:
            sel = os.path.join(ROOT, TREE, 'selected', gp_dir, algo, 'SELECTED_SEED.txt')
            if not os.path.exists(sel):
                missing.append(f'{gp_dir}/{algo}'); continue
            txt = open(sel).read().strip()
            seed = int(re.match(r'seed (\d+)', txt).group(1))
            t = tasks[f'v6__seed{seed:02d}__{gp_dir}__{algo}']
            rows.append(dict(scope='multi_driver', model_path=f'{TREE[len("trained_models/"):]}/selected/{gp_dir}/{algo}',
                             gp=t['gp'], driver=t['driver'], algorithm=algo, effective_seed=seed,
                             script='scripts/rival_dnf_v6_20261014.py ' + t['script'],
                             args=template(t['args']), env=ENV,
                             notes=f'v6; selected among seeds 1-10 on 200 validation races ({txt})'))
    if missing:
        print('sin modelo elegido:', missing)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} filas -> {OUT}')


if __name__ == '__main__':
    main()
