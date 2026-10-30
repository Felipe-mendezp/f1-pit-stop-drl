"""
2026-10-30 — Piloto: reentrenar solo Hungarian (TSU) con ruido de la Ec. 2 (scripts/eq2_noise_20261030.py, sd 0.28).
50 tareas = 5 algoritmos x 10 semillas, con la misma configuracion que los agentes elegidos de v10eGQf:
PPO con ent_coef 0.05 (como tasks_v10eGQf_ppoent05), RPPO/DQN/A2C/TRPO como tasks_v10eGQf_20261029.
Salida: trained_models/retrained_v10eGQfq_20261030.
"""
import os
here = os.path.dirname(os.path.abspath(__file__))
OLD, NEW = 'retrained_v10eGQf_20261029', 'retrained_v10eGQfq_20261030'
out = []
for line in open(os.path.join(here, 'tasks_v10eGQf_20261029.tsv')).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    if '__Hungarian_' not in name:
        continue
    if name.endswith('__PPO'):
        assert '--ent-coef 0.02' in cmd; cmd = cmd.replace('--ent-coef 0.02', '--ent-coef 0.05')
    assert cmd.startswith('scripts/overtake_v10f_20261029.py ')
    cmd = 'scripts/eq2_noise_20261030.py ' + cmd.replace(OLD, NEW)
    out.append('\t'.join([name.replace('v10eGQf__', 'v10eGQfq__'), log.replace(OLD, NEW).replace('v10eGQf__', 'v10eGQfq__'), sim, cmd]))
p = os.path.join(here, 'tasks_v10eGQfq_hun_20261030.tsv')
open(p, 'w').write('\n'.join(out) + '\n'); print(p, len(out))
