"""2026-10-04 — Entrena PPO y RecurrentPPO de los 8 GP del paper (sin Emilia Romagna) sobre los
simuladores con coeficientes de neumatico alternativos (scripts/build_simulation_tyrecoef_20261004.py).

Para cada opcion y cada fila multi_driver PPO/RecurrentPPO de reproduce/MODELS_MANIFEST.csv se
replica la llamada del manifiesto (mismo script, argumentos, piloto y semilla efectiva) con:
    F1_SIMULATION_DIR = trained_models/simulation_tyre_<opcion>_20261004
    F1_LASTLAP_PITOUT = 1   (fisica final del paper para todas las corridas nuevas)
    salida            = trained_models/rl_agents_tyre_<opcion>_20261004
Las corridas que ya tienen su modelo final se saltan (reanudable). Log por corrida en <salida>/logs/.
Opcion 'control' (2026-10-05): mismo lanzador y semillas sobre el simulador PUBLICADO
(trained_models/simulation, solo lectura) -> mide el ruido de reentrenamiento.

    python scripts/run_tyre_options_training_20261004.py [--options opcion2 opcion3] [--jobs 3]
        [--only Singapore] [--timesteps N --out-suffix _smoke] [--dry-run]
"""
import argparse, csv, os, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--options', nargs='+', default=['opcion2', 'opcion3'])
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--only', default=None)
    ap.add_argument('--timesteps', type=int, default=None)
    ap.add_argument('--out-suffix', default='')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    py = os.environ.get('PYTHON', sys.executable)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST.csv')))
            if r['scope'] == 'multi_driver' and r['algorithm'] in ('PPO', 'RecurrentPPO')
            and 'Emilia_Romagna' not in r['model_path'] and (a.only is None or a.only in r['model_path'])]
    jobs = []
    for opt in a.options:
        sim = os.path.join(ROOT, 'trained_models', 'simulation' if opt == 'control'
                           else f'simulation_tyre_{opt}_20261004')
        out = os.path.join(ROOT, 'trained_models', f'rl_agents_tyre_{opt}_20261004{a.out_suffix}')
        assert os.path.isdir(sim), sim
        os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
        for r in rows:
            args = r['args'].replace('{PHYS}', PHYS).replace('{OUT}', shlex.quote(out))
            if a.timesteps:
                args += f' --timesteps {a.timesteps}'
            cmd = [py, os.path.join(ROOT, r['script'])] + shlex.split(args)
            env = dict(os.environ, OMP_NUM_THREADS='1', F1_SIMULATION_DIR=sim, F1_DSQ_PENALTY='-1000',
                       F1_LASTLAP_PITOUT='1', F1_SC_BUNCHING='1', F1_SC_BUNCH_DELTA='1.5',
                       F1_FOLLOWING_GAP_SCALE='1.0')
            env.pop('F1_RL_AGENTS_DIR', None)
            name = f"{opt}__{r['model_path'].split('/', 1)[1].replace('/', '__')}"
            jobs.append((name, cmd, env, os.path.join(out, 'logs', name + '.log')))
    print(f'{len(jobs)} training call(s), {a.jobs} in parallel')

    def run(job):
        name, cmd, env, log = job
        if a.dry_run:
            print(name, ' '.join(shlex.quote(c) for c in cmd)); return name, 'dry'
        if os.path.isfile(log + '.done'):
            return name, 'skip (done)'
        t0 = time.time()
        with open(log, 'w') as fh:
            rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            open(log + '.done', 'w').write(f'{time.time() - t0:.0f} s\n')
        msg = f'rc={rc} {(time.time() - t0) / 60:.1f} min'
        print(time.strftime('%H:%M'), name, msg, flush=True)
        return name, msg

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, jobs))
    bad = [n for n, m in res if m.startswith('rc=') and not m.startswith('rc=0')]
    print('FAILED:', bad if bad else 'none')


if __name__ == '__main__':
    main()
