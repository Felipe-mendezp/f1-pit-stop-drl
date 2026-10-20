"""Manifest de procedencia de los agentes v8 (version del paper) — 2026-10-20.

Una fila por agente usado en el paper, con las mismas columnas que reproduce/MODELS_MANIFEST.csv (v2.0.0, que no se
modifica):
  - multi_driver (Figs. 6-10): trained_models/rl_agents_v8_20261018/selected/<GP_piloto>/<algo>; comando de
    reproduce/cluster/tasks_v8.tsv (tarea de la semilla elegida) y semilla de SELECTED_SEED.txt
    (scripts/select_seeds_v5_20261010.py --calib v8, reglas 5/7/8 y --max-dsq 0.01).
  - single_driver (Fig. 5, Tabla N.1): trained_models/rl_agents_onedriver_v8_20261019/seedXX/one_driver/<GP>/<algo>
    con la semilla de results/onedriver_v8_20261019/selection_20261019.tsv (menor brecha con la QMIP); comando de
    reproduce/cluster/tasks_onedriver_v8.tsv.

    python scripts/make_manifest_v8_20261020.py            # -> reproduce/MODELS_MANIFEST_v8_20261020.csv
    python scripts/make_manifest_v8_20261020.py --check    # verifica que cada fila apunta a un best_model.zip
"""
import os, re, csv, shlex, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
OUT = os.path.join(ROOT, 'reproduce', 'MODELS_MANIFEST_v8_20261020.csv')
MULTI = 'trained_models/rl_agents_v8_20261018'
SINGLE = 'trained_models/rl_agents_onedriver_v8_20261019'
SIM = 'trained_models/simulation_v7_20261015'
SIM1 = 'trained_models/simulation_onedriver_v7_20261015'
WRAP = 'scripts/rival_start_v8_20261018.py'
PHYS = '--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5'
ENV_MULTI = (f'source reproduce/env_paper_v3_20261007.sh; F1_SIMULATION_DIR={SIM}; F1_RIVAL_DNF, v6 drivers, '
             'F1_RIVAL_START_ALPHA (data/rival_start_alpha_2021_2023_20261018.json), F1_LAP0_ACTION0_MEDIUM=1 and '
             'F1_LAP0_PENALTY=-1000 set by the wrapper')
ENV_SINGLE = (f'source reproduce/env_paper_v3_20261007.sh; simulator {SIM1}; F1_ONE_DRIVER_* penalties '
              '(lap 0 -1000 and medium, consecutive -200, laps 1-5 -100, last lap -150, DSQ -1000) set by the wrapper')
COLS = ['scope', 'model_path', 'gp', 'driver', 'algorithm', 'effective_seed', 'script', 'args', 'env', 'notes']
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO']


def read_tsv(name):
    out = {}
    for line in open(os.path.join(CL, name)):
        if line.strip():
            n, _log, sim, cmd = line.rstrip('\n').split('\t')
            out[n] = (sim, shlex.split(cmd))
    return out


def template(args):
    s = ' '.join(shlex.quote(a) if ' ' in a else a for a in args).replace("'", '"')
    s = s.replace(PHYS, '{PHYS}')
    return re.sub(r'(--out-root|--output-dir|--agents-dir) \S+', r'\1 {OUT}', s)


def multi_rows():
    tasks, rows = read_tsv('tasks_v8.tsv'), []
    for gp_dir in sorted({n.split('__')[2] for n in tasks}):
        for algo in ALGOS:
            txt = open(os.path.join(ROOT, MULTI, 'selected', gp_dir, algo, 'SELECTED_SEED.txt')).read().strip()
            seed = int(re.match(r'seed (\d+)', txt).group(1))
            sim, tok = tasks[f'v8__seed{seed:02d}__{gp_dir}__{algo}']
            assert sim == SIM and tok[0] == WRAP, (gp_dir, algo)
            args = tok[2:]
            gp = args[args.index('--gp' if '--gp' in args else '--gps') + 1]
            rows.append(dict(scope='multi_driver', model_path=f'{MULTI[15:]}/selected/{gp_dir}/{algo}', gp=gp,
                             driver='Driver_' + gp_dir.rsplit('_', 1)[1], algorithm=algo, effective_seed=seed,
                             script=f'{WRAP} {tok[1]}', args=template(args), env=ENV_MULTI,
                             notes=f'v8; selected among seeds 1-10 on 200 validation races ({txt})'))
    return rows


def single_rows():
    tasks, rows = read_tsv('tasks_onedriver_v8.tsv'), []
    sel = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'onedriver_v8_20261019', 'selection_20261019.tsv')),
                              delimiter='\t'))
    for r in sel:
        g = r['gp'].replace(' Grand Prix', ''); seed = int(r['best_seed'])
        sim, tok = tasks[f"onedriver_v8__seed{seed:02d}__{g.replace(' ', '_')}__{r['algo']}"] \
            if f"onedriver_v8__seed{seed:02d}__{g.replace(' ', '_')}__{r['algo']}" in tasks \
            else tasks[f"onedriver_v8__seed{seed:02d}__{g}__{r['algo']}"]
        assert sim == SIM1 and tok[0] == 'scripts/one_driver_v8_20261019.py', r
        rows.append(dict(scope='single_driver', model_path=f'{SINGLE[15:]}/seed{seed:02d}/one_driver/{g}/{r["algo"]}',
                         gp=r['gp'], driver='VER', algorithm=r['algo'], effective_seed=seed,
                         script=f'{tok[0]} {tok[1]}', args=template(tok[2:]), env=ENV_SINGLE,
                         notes=f'v8; seed with the smallest gap to the QMIP among seeds 1-5 '
                               f'(gap {r["best_gap"]} s; all {r["n_sane"]} seeds sane)'))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    rows = multi_rows() + single_rows()
    miss = [r['model_path'] for r in rows
            if not os.path.exists(os.path.join(ROOT, 'trained_models', r['model_path'], 'best_model.zip'))]
    print(f"{sum(r['scope'] == 'multi_driver' for r in rows)} multi_driver + "
          f"{sum(r['scope'] == 'single_driver' for r in rows)} single_driver; sin best_model.zip: {miss or 'ninguno'}")
    if a.check:
        return
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} filas -> {OUT}')


if __name__ == '__main__':
    main()
