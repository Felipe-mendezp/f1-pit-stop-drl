"""2026-10-18 — Tareas de entrenamiento v8 (v7 + compuestos de partida de rivales Dirichlet-multinomial).

Las mismas 400 tareas de reproduce/cluster/tasks_v7.tsv (8 GP, 5 algoritmos x 10 semillas, mismos hiperparametros,
semillas y simulador trained_models/simulation_v7_20261015), lanzadas con scripts/rival_start_v8_20261018.py en vez de
scripts/rival_dnf_v6_20261014.py y con salida trained_models/rl_agents_v8_20261018/.
Piloto: Bahrain TRPO, Dutch A2C y United States A2C x 10 semillas (30 tareas, subconjunto de tasks_v8.tsv).
2026-10-18 (noche): ademas tasks_v8_rest.tsv = las 370 que no son del piloto (el piloto se lanzo solo; asi no se repite).

    python scripts/make_tasks_v8_20261018.py  ->  reproduce/cluster/tasks_v8.tsv, tasks_v8_pilot.tsv, tasks_v8_rest.tsv
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(ROOT, 'reproduce', 'cluster')
O7, O8 = 'trained_models/rl_agents_v7_20261015', 'trained_models/rl_agents_v8_20261018'
W7, W8 = 'scripts/rival_dnf_v6_20261014.py', 'scripts/rival_start_v8_20261018.py'
PILOT = [('Bahrain_ALO', 'TRPO'), ('Dutch_STR', 'A2C'), ('United_States_MAG', 'A2C')]


def main():
    lines = []
    for line in open(os.path.join(CL, 'tasks_v7.tsv')).read().splitlines():
        name, log, sim, cmd = line.split('\t')
        assert name.startswith('v7__') and O7 in log and O7 in cmd and cmd.startswith(W7 + ' '), line
        name = 'v8__' + name[len('v7__'):]
        log = log.replace(O7, O8).replace('/v7__', '/v8__')
        cmd = W8 + cmd[len(W7):].replace(O7, O8)
        lines.append('\t'.join([name, log, sim, cmd]))
    assert len(lines) == 400 and len({l.split('\t')[0] for l in lines}) == 400
    assert not any(O7 in l for l in lines)
    pilot = [l for l in lines if tuple(l.split('\t')[0].split('__')[2:4]) in PILOT]
    assert len(pilot) == 30, len(pilot)
    rest = [l for l in lines if l not in pilot]
    assert len(rest) == 370
    for fname, ls in [('tasks_v8.tsv', lines), ('tasks_v8_pilot.tsv', pilot), ('tasks_v8_rest.tsv', rest)]:
        with open(os.path.join(CL, fname), 'w') as f:
            f.write('\n'.join(ls) + '\n')
        print(f'{fname}: {len(ls)} tareas')


if __name__ == '__main__':
    main()
