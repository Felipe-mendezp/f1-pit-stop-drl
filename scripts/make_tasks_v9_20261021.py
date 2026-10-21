"""2026-10-21 — Tareas de entrenamiento v9 (v8 + juegos de neumaticos reales de 2024, scripts/tyre_sets_v9_20261021.py).

Multi-piloto: las mismas 400 tareas de reproduce/cluster/tasks_v8.tsv (8 GP, 5 algoritmos x 10 semillas, mismos
hiperparametros, semillas y simulador trained_models/simulation_v7_20261015), anteponiendo el envoltorio v9 y con salida
trained_models/rl_agents_v9_20261021/. Piloto = Bahrain TRPO, Dutch A2C y United States A2C x 10 semillas (30, como v8).
Mono-piloto: las 200 tareas de reproduce/cluster/tasks_onedriver_v8.tsv con el envoltorio v9 y salida
trained_models/rl_agents_onedriver_v9_20261021/.

    python scripts/make_tasks_v9_20261021.py  ->  reproduce/cluster/tasks_v9.tsv, tasks_v9_pilot.tsv, tasks_v9_rest.tsv,
                                                   tasks_onedriver_v9.tsv
2026-10-21 (noche): inventario por reglamento (data/tyre_sets_2024_reg_20261021.csv) -> etiqueta v9r, para no mezclar
con los modelos mono-piloto v9 ya lanzados con el inventario anterior (rl_agents_onedriver_v9_20261021, cancelados):
    python scripts/make_tasks_v9_20261021.py v9r  ->  tasks_v9r.tsv, tasks_onedriver_v9r.tsv
                                                     (salidas rl_agents_v9r_20261021, rl_agents_onedriver_v9r_20261021)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
W9 = 'scripts/tyre_sets_v9_20261021.py'
PILOT = [('Bahrain_ALO', 'TRPO'), ('Dutch_STR', 'A2C'), ('United_States_MAG', 'A2C')]


def convert(src, old_tag, new_tag, old_dir, new_dir, n):
    lines = []
    for line in open(os.path.join(CL, src)).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert name.startswith(old_tag + '__') and old_dir in log and old_dir in cmd, line
        name = new_tag + name[len(old_tag):]
        log = log.replace(old_dir, new_dir).replace(f'/{old_tag}__', f'/{new_tag}__')
        cmd = W9 + ' ' + cmd.replace(old_dir, new_dir)
        lines.append('\t'.join([name, log, sim, cmd]))
    assert len(lines) == n and len({l.split('\t')[0] for l in lines}) == n
    assert not any(old_dir in l or f'{old_tag}__' in l for l in lines)
    return lines


def write(fname, ls):
    with open(os.path.join(CL, fname), 'w') as f:
        f.write('\n'.join(ls) + '\n')
    print(f'{fname}: {len(ls)} tareas')


def main(tag='v9'):
    multi = convert('tasks_v8.tsv', 'v8', tag, 'trained_models/rl_agents_v8_20261018',
                    f'trained_models/rl_agents_{tag}_20261021', 400)
    one = convert('tasks_onedriver_v8.tsv', 'onedriver_v8', f'onedriver_{tag}',
                  'trained_models/rl_agents_onedriver_v8_20261019', f'trained_models/rl_agents_onedriver_{tag}_20261021', 200)
    write(f'tasks_{tag}.tsv', multi)
    write(f'tasks_onedriver_{tag}.tsv', one)
    if tag == 'v9':
        pilot = [l for l in multi if tuple(l.split('\t')[0].split('__')[2:4]) in PILOT]
        assert len(pilot) == 30, len(pilot)
        write('tasks_v9_pilot.tsv', pilot)
        write('tasks_v9_rest.tsv', [l for l in multi if l not in pilot])


if __name__ == '__main__':
    main(*sys.argv[1:2])
