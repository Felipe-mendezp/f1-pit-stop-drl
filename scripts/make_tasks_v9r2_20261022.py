"""2026-10-22 — Tareas de entrenamiento v9r2 (v9r + juegos reutilizables, scripts/tyre_reuse_v9r2_20261022.py).

Las mismas tareas de reproduce/cluster/tasks_v9r.tsv (multi-piloto, 400) y tasks_onedriver_v9r.tsv (mono-piloto, 200)
(mismos GP, algoritmos, semillas, hiperparametros y simuladores), anteponiendo el envoltorio v9r2 y con salida
trained_models/rl_agents_v9r2_20261022/ y trained_models/rl_agents_onedriver_v9r2_20261022/.
    python scripts/make_tasks_v9r2_20261022.py  ->  reproduce/cluster/tasks_v9r2.tsv, tasks_onedriver_v9r2.tsv
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
W = 'scripts/tyre_reuse_v9r2_20261022.py'


def convert(src, old_tag, new_tag, old_dir, new_dir, n):
    lines = []
    for line in open(os.path.join(CL, src)).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert name.startswith(old_tag + '__') and old_dir in log and old_dir in cmd, line
        assert cmd.startswith('scripts/tyre_sets_v9_20261021.py '), cmd
        name = new_tag + name[len(old_tag):]
        log = log.replace(old_dir, new_dir).replace(f'/{old_tag}__', f'/{new_tag}__')
        cmd = W + ' ' + cmd.replace(old_dir, new_dir)
        lines.append('\t'.join([name, log, sim, cmd]))
    assert len(lines) == n and len({l.split('\t')[0] for l in lines}) == n
    assert not any(old_dir in l or f'{old_tag}__' in l for l in lines)
    return lines


def write(fname, ls):
    with open(os.path.join(CL, fname), 'w') as f:
        f.write('\n'.join(ls) + '\n')
    print(f'{fname}: {len(ls)} tareas')


if __name__ == '__main__':
    write('tasks_v9r2.tsv', convert('tasks_v9r.tsv', 'v9r', 'v9r2', 'trained_models/rl_agents_v9r_20261021',
                                    'trained_models/rl_agents_v9r2_20261022', 400))
    write('tasks_onedriver_v9r2.tsv', convert('tasks_onedriver_v9r.tsv', 'onedriver_v9r', 'onedriver_v9r2',
                                              'trained_models/rl_agents_onedriver_v9r_20261021',
                                              'trained_models/rl_agents_onedriver_v9r2_20261022', 200))
