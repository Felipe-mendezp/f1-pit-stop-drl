"""
2026-10-30 — v10g (= v10eGQf + ruido de la Ec. 2). Piloto: reentrenar solo Hungarian (TSU) con ruido de la Ec. 2 (scripts/eq2_noise_20261030.py, sd 0.28).
50 tareas = 5 algoritmos x 10 semillas, con la misma configuracion que los agentes elegidos de v10eGQf:
PPO con ent_coef 0.05 (como tasks_v10eGQf_ppoent05), RPPO/DQN/A2C/TRPO como tasks_v10eGQf_20261029.
Salida: trained_models/retrained_v10g_20261030.
Variante 'ppo8': solo PPO (ent 0.05) en los 8 GP, 80 tareas -> tasks_v10g_ppo8_20261030.tsv.
"""
import os
here = os.path.dirname(os.path.abspath(__file__))
OLD, NEW = 'retrained_v10eGQf_20261029', 'retrained_v10g_20261030'
for tag, keep in [('hun', lambda n: '__Hungarian_' in n), ('ppo8', lambda n: n.endswith('__PPO'))]:
  out = []
  for line in open(os.path.join(here, 'tasks_v10eGQf_20261029.tsv')).read().splitlines():
      name, log, sim, cmd = line.split('\t')
      if not keep(name):
          continue
      if name.endswith('__PPO'):
          assert '--ent-coef 0.02' in cmd; cmd = cmd.replace('--ent-coef 0.02', '--ent-coef 0.05')
      assert cmd.startswith('scripts/overtake_v10f_20261029.py ')
      cmd = 'scripts/eq2_noise_20261030.py ' + cmd.replace(OLD, NEW)
      out.append('\t'.join([name.replace('v10eGQf__', 'v10g__'), log.replace(OLD, NEW).replace('v10eGQf__', 'v10g__'), sim, cmd]))
  p = os.path.join(here, f'tasks_v10g_{tag}_20261030.tsv')
  open(p, 'w').write('\n'.join(out) + '\n'); print(p, len(out))
