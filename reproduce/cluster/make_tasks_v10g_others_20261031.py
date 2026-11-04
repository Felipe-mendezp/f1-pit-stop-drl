"""
2026-10-31 — v10g, los 4 metodos DRL restantes (DQN, RecurrentPPO, A2C, TRPO) en los 8 GP.
320 tareas = 4 algoritmos x 8 GP x 10 semillas, misma configuracion que tasks_v10eGQf_20261029
(PPO ya corrio aparte en tasks_v10g_ppo8_20261030.tsv con ent_coef 0.05).
RPPO: variante 'base' = sin --ent-coef (como v10eGQf); variante 'rppo05' = --ent-coef 0.05
(como el reentrenamiento de Miami del 29-oct, tasks_v10eGQf_rppoent05_20261029.tsv).
Salida: trained_models/retrained_v10g_20261030 (el mismo arbol que PPO).
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
OLD, NEW = 'retrained_v10eGQf_20261029', 'retrained_v10g_20261030'
variant = sys.argv[1] if len(sys.argv) > 1 else 'base'
assert variant in ('base', 'rppo05'), variant
out = []
for line in open(os.path.join(here, 'tasks_v10eGQf_20261029.tsv')).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    if name.endswith('__PPO'):
        continue
    if variant == 'rppo05' and name.endswith('__RecurrentPPO'):
        assert '--hyperparams' not in cmd
        cmd = cmd.replace('--algorithms RecurrentPPO',
                          '--algorithms RecurrentPPO --hyperparams rppoent05_20261029')
        assert '--hyperparams rppoent05_20261029' in cmd
    assert cmd.startswith('scripts/overtake_v10f_20261029.py ')
    cmd = 'scripts/eq2_noise_20261030.py ' + cmd.replace(OLD, NEW)
    out.append('\t'.join([name.replace('v10eGQf__', 'v10g__'),
                          log.replace(OLD, NEW).replace('v10eGQf__', 'v10g__'), sim, cmd]))
p = os.path.join(here, f'tasks_v10g_others_{variant}_20261031.tsv')
open(p, 'w').write('\n'.join(out) + '\n'); print(p, len(out))
