"""
2026-11-02 - 49 tareas mas de post-entrenamiento startmix: semillas 4 a 10 en los siete GP distintos de Miami, para
que las ocho carreras queden con 10 semillas igual que el entrenamiento original (decision del autor: tener 10
semillas solo en Miami se ve arbitrario en el apendice).
Derivadas de las lineas de semilla 1 de tasks_v10gsm_20261101.tsv: identicas salvo --seed, el nombre, el log y
--out-root. Mismo arbol de salida (trained_models/retrained_v10g_startmix_20261101), asi que las semillas 1 a 3, las
4 a 10 de Miami y la carpeta selected/ no se tocan.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'tasks_v10gsm_20261101.tsv')
LINES = [l for l in open(SRC).read().splitlines() if l.startswith('v10gsm__seed01__')]
assert len(LINES) == 8, len(LINES)

out = []
for line in LINES:
    name, log, sim, cmd = line.split('\t')
    gp = name.split('__')[2]
    if gp == 'Miami_OCO':
        continue
    assert '--algo PPO' in cmd and '--lr 1e-4' in cmd and '--start-mix 0.5' in cmd
    assert '--timesteps 1500000' in cmd and '--following-gap-scale 0.26' in cmd
    assert '--seed 1 ' in cmd
    assert '--agents trained_models/retrained_v10g_20261030/selected' in cmd
    assert cmd.endswith('--out-root trained_models/retrained_v10g_startmix_20261101/seed01')
    assert sim == 'trained_models/simulation_v10eGQ_20261028', sim
    for s in range(4, 11):
        n = name.replace('seed01', f'seed{s:02d}')
        l = log.replace('seed01', f'seed{s:02d}')
        c = cmd.replace('--seed 1 ', f'--seed {s} ').replace('/seed01', f'/seed{s:02d}')
        assert c.count(f'--seed {s} ') == 1 and c.endswith(f'/seed{s:02d}')
        assert len(re.findall(r'seed01', c)) == 0
        out.append('\t'.join([n, l, sim, c]))
assert len(out) == 49, len(out)
p = os.path.join(HERE, 'tasks_v10gsm_rest_20261102.tsv')
open(p, 'w').write('\n'.join(out) + '\n')
print(p, len(out))
