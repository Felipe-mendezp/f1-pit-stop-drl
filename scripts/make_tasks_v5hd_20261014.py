"""2026-10-14 — Tareas multi-piloto v5hd: los 4 GP con piloto nuevo (scripts/gp_drivers_v5hd_20261014.py),
5 algoritmos x 10 semillas = 200 tareas. Mismos comandos que tasks_v5h8_ppo.tsv y tasks_v5h8_others.tsv (simulador
trained_models/simulation_v5h8_20261012, mismos hiperparametros y semillas), con tres cambios:
  * cada comando pasa por el envoltorio gp_drivers_v5hd_20261014.py (cambia GP_CONFIGS en memoria);
  * --driver <anterior> -> --driver <nuevo> (A2C, TRPO, RecurrentPPO; PPO y DQN toman el piloto de GP_CONFIGS);
  * salida trained_models/rl_agents_v5hd_20261014/seed<ss>/<GP_piloto nuevo>/<algo>/.
Los otros 4 GP (Bahrain, Saudi, Miami, Singapore) siguen en trained_models/rl_agents_v5h_20261011.
    python scripts/make_tasks_v5hd_20261014.py   ->  reproduce/cluster/tasks_v5hd.tsv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from gp_drivers_v5hd_20261014 import CHANGES
CL = os.path.join(ROOT, 'reproduce', 'cluster')
O0, O1 = 'trained_models/rl_agents_v5h_20261011', 'trained_models/rl_agents_v5hd_20261014'
WRAP = 'scripts/gp_drivers_v5hd_20261014.py'


def main():
    lines = []
    for src in ('tasks_v5h8_ppo.tsv', 'tasks_v5h8_others.tsv'):
        for line in open(os.path.join(CL, src)).read().splitlines():
            name, log, sim, cmd = line.split('\t')
            gp = next((g for g in CHANGES if f"'{g}'" in cmd), None)
            if gp is None:
                continue
            new, _, old, _ = CHANGES[gp]
            short = gp.replace(' Grand Prix', '').replace(' ', '_')
            a, b = f'{short}_{old[7:]}', f'{short}_{new[7:]}'
            assert name.startswith('v5h__') and a in name and O0 in log and O0 in cmd, line
            name = 'v5hd__' + name[len('v5h__'):].replace(a, b)
            log = log.replace(O0, O1).replace('v5h__', 'v5hd__').replace(a, b)
            cmd = WRAP + ' ' + cmd.replace(O0, O1).replace(f'--driver {old}', f'--driver {new}')
            assert old not in cmd, cmd
            lines.append('\t'.join([name, log, sim, cmd]))
    assert len(lines) == 200, len(lines)
    path = os.path.join(CL, 'tasks_v5hd.tsv')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{path}: {len(lines)} tareas')


if __name__ == '__main__':
    main()
