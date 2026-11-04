"""
2026-11-02 - 7 tareas mas de post-entrenamiento startmix en Miami (semillas 4 a 10), para completar las 10 semillas
del caso original. Motivo: con las 3 primeras, la elegida por la regla degrada el escenario de la Fig. 7 en 1,55
posiciones y lleva el promedio de los 8 GP a +0,141, sobre el criterio de 0,1
(results/startmix_v10g_20261101/e3_compare_20261102.csv).
Derivado de la linea de Miami semilla 1 de tasks_v10gsm_20261101.tsv: identico salvo --seed, el nombre, el log y
--out-root. Mismo arbol de salida (trained_models/retrained_v10g_startmix_20261101), asi que las semillas 1 a 3 y la
carpeta selected/ no se tocan.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'tasks_v10gsm_20261101.tsv')
BASE = [l for l in open(SRC).read().splitlines() if l.startswith('v10gsm__seed01__Miami_OCO__PPO\t')]
assert len(BASE) == 1, BASE
name, log, sim, cmd = BASE[0].split('\t')
assert "--gp 'Miami Grand Prix'" in cmd and '--seed 1 ' in cmd
assert '--algo PPO' in cmd and '--lr 1e-4' in cmd and '--start-mix 0.5' in cmd
assert '--timesteps 1500000' in cmd and '--following-gap-scale 0.26' in cmd
assert '--agents trained_models/retrained_v10g_20261030/selected' in cmd
assert cmd.endswith('--out-root trained_models/retrained_v10g_startmix_20261101/seed01')
assert sim == 'trained_models/simulation_v10eGQ_20261028', sim

out = []
for s in range(4, 11):
    n = name.replace('seed01', f'seed{s:02d}')
    l = log.replace('seed01', f'seed{s:02d}')
    c = cmd.replace('--seed 1 ', f'--seed {s} ').replace('/seed01', f'/seed{s:02d}')
    assert c.count(f'--seed {s} ') == 1 and c.endswith(f'/seed{s:02d}')
    assert len(re.findall(r'seed01', c)) == 0
    out.append('\t'.join([n, l, sim, c]))
p = os.path.join(HERE, 'tasks_v10gsm_miami_20261102.tsv')
open(p, 'w').write('\n'.join(out) + '\n')
print(p, len(out))
