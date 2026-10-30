"""
2026-10-29 — Tareas de reentrenamiento con más entropía (opción B del diagnóstico de Miami, OK del autor 29-oct):
  - tasks_v10eGQf_ppoent05_20261029.tsv: PPO, ent_coef 0.05, 8 GP x 10 semillas (80 tareas);
  - tasks_v10eGQf_rppoent05_20261029.tsv: RecurrentPPO, ent_coef 0.05 (hyperparams/config_rppoent05_20261029.py),
    solo Miami, 10 semillas (prueba antes de decidir RPPO en los 8 GP).
Se derivan de tasks_v10eGQf_20261029.tsv: mismo comando, solo cambia la entropía y el árbol de salida.
"""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'tasks_v10eGQf_20261029.tsv')).read().splitlines()
OLD = 'retrained_v10eGQf_20261029'
specs = [
    ('ppoent05', lambda n: n.endswith('__PPO'), 'retrained_v10eGQf_ppoent05_20261029',
     lambda c: c.replace('--ent-coef 0.02', '--ent-coef 0.05')),
    ('rppoent05', lambda n: n.endswith('__RecurrentPPO') and '__Miami_' in n, 'retrained_v10eGQf_rppoent05_20261029',
     lambda c: c.replace('--algorithms RecurrentPPO', '--algorithms RecurrentPPO --hyperparams rppoent05_20261029')),
]
for tag, keep, new, edit in specs:
    out = []
    for line in src:
        name, log, sim, cmd = line.split('\t')
        if not keep(name):
            continue
        cmd2 = edit(cmd).replace(OLD, new)
        assert cmd2 != cmd.replace(OLD, new), name
        out.append('\t'.join([name.replace('v10eGQf__', f'v10eGQf{tag}__'), log.replace(OLD, new)
                              .replace('v10eGQf__', f'v10eGQf{tag}__'), sim, cmd2]))
    p = os.path.join(here, f'tasks_v10eGQf_{tag}_20261029.tsv')
    open(p, 'w').write('\n'.join(out) + '\n')
    print(p, len(out))
