"""2026-10-14 — Tareas multi-piloto v6: los 8 GP (pilotos v5hd) con abandonos de rivales en el simulador,
5 algoritmos x 10 semillas = 400 tareas. Mismos comandos, hiperparametros y semillas que tasks_v5h8_ppo.tsv,
tasks_v5h8_others.tsv y tasks_v5h_saudi.tsv, con estos cambios:
  * cada comando pasa por scripts/rival_dnf_v6_20261014.py (F1_RIVAL_DNF + pilotos v5hd);
  * en los 4 GP con piloto nuevo, --driver <anterior> -> --driver <nuevo> y carpeta <GP_piloto nuevo>;
  * simulador trained_models/simulation_v6_20261014 en los 8 GP (v5h8 + Ec. 2 sin Australia 2024; Bahrain y Saudi identicos);
  * orden: primero Saudi y Singapore (piloto de la correccion);
  * salida trained_models/rl_agents_v6_20261014/seed<ss>/<GP_piloto>/<algo>/.
    python scripts/make_tasks_v6_20261014.py   ->  reproduce/cluster/tasks_v6.tsv
"""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from gp_drivers_v5hd_20261014 import CHANGES
CL = os.path.join(ROOT, 'reproduce', 'cluster')
O0, O1 = 'trained_models/rl_agents_v5h_20261011', 'trained_models/rl_agents_v6_20261014'
SIM = 'trained_models/simulation_v6_20261014'
WRAP = 'scripts/rival_dnf_v6_20261014.py'


def main():
    lines = []
    for src in ('tasks_v5h8_ppo.tsv', 'tasks_v5h8_others.tsv', 'tasks_v5h_saudi.tsv'):
        for line in open(os.path.join(CL, src)).read().splitlines():
            name, log, sim, cmd = line.split('\t')
            assert name.startswith('v5h__') and O0 in log and O0 in cmd, line
            name, log, cmd = 'v6__' + name[len('v5h__'):], log.replace(O0, O1).replace('v5h__', 'v6__'), cmd.replace(O0, O1)
            gp = next((g for g in CHANGES if f"'{g}'" in cmd), None)
            if gp is not None:
                new, _, old, _ = CHANGES[gp]
                short = gp.replace(' Grand Prix', '').replace(' ', '_')
                a, b = f'{short}_{old[7:]}', f'{short}_{new[7:]}'
                name, log = name.replace(a, b), log.replace(a, b)
                cmd = cmd.replace(f'--driver {old}', f'--driver {new}')
                assert old not in cmd, cmd
            lines.append('\t'.join([name, log, SIM, WRAP + ' ' + cmd]))
    # primero Saudi y Singapore (los GP donde los abandonos gatillaban paradas raras): sus resultados sirven de
    # piloto para verificar que v6 corrige el problema antes de que termine el resto
    first = ("'Saudi Arabian Grand Prix'", "'Singapore Grand Prix'")
    lines.sort(key=lambda l: 0 if any(g in l for g in first) else 1)
    assert len(lines) == 400 and len(set(l.split('\t')[0] for l in lines)) == 400, len(lines)
    assert not any('rl_agents_v5h_' in l for l in lines)
    path = os.path.join(CL, 'tasks_v6.tsv')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{path}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
