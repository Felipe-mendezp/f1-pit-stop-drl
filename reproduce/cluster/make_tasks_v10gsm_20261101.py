"""
2026-11-01 - Post-entrenamiento con largada sorteada (startmix) de PPO sobre los agentes v10g.
24 tareas = 8 GP x 3 semillas. Igual a tasks_v9r2sm.tsv (PPO, --lr 1e-4, --start-mix 0.5, 1.5M pasos),
pero con la fisica de v10g: arbol de simulacion simulation_v10eGQ_20261028, los envoltorios de v10g
(ruido de la Ec. 2, adelantamiento v10f, edad efectiva de los neumaticos) y --following-gap-scale 0.26.
El ent_coef 0.05 de PPO v10g viene en best_model.zip y lo hereda finetune_startmix (lo imprime al partir).
Salida: trained_models/retrained_v10g_startmix_20261101.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_TREE, NEW_TREE = 'retrained_v9r2_startmix_20261025', 'retrained_v10g_startmix_20261101'
OLD_SIM, NEW_SIM = 'trained_models/simulation_v7_20261015', 'trained_models/simulation_v10eGQ_20261028'
OLD_AGENTS = '--agents trained_models/rl_agents_v9r2_20261022/selected'
NEW_AGENTS = '--agents trained_models/retrained_v10g_20261030/selected'
V10G_WRAPPERS = ('scripts/eq2_noise_20261030.py scripts/overtake_v10f_20261029.py '
                 'scripts/tyre_ages_v10eGQ_20261028.py scripts/v10_wrapper_20261027.py ')
V9R2_WRAPPERS = ('scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py '
                 'scripts/rival_start_v8_20261018.py scripts/finetune_startmix_20261025.py ')

out = []
for line in open(os.path.join(HERE, 'tasks_v9r2sm.tsv')).read().splitlines():
    name, log, sim, cmd = line.split('\t')
    if not name.endswith('__PPO'):
        continue
    assert sim == OLD_SIM, sim
    assert cmd.startswith(V9R2_WRAPPERS), cmd[:120]
    assert OLD_AGENTS in cmd and '--following-gap-scale 1.0' in cmd and '--lr 1e-4' in cmd
    cmd = V10G_WRAPPERS + cmd
    cmd = cmd.replace(OLD_AGENTS, NEW_AGENTS)
    cmd = cmd.replace('--following-gap-scale 1.0', '--following-gap-scale 0.26')
    cmd = cmd.replace(OLD_TREE, NEW_TREE)
    assert OLD_TREE not in cmd
    out.append('\t'.join([name.replace('v9r2sm__', 'v10gsm__'),
                          log.replace(OLD_TREE, NEW_TREE).replace('v9r2sm__', 'v10gsm__'),
                          NEW_SIM, cmd]))
p = os.path.join(HERE, 'tasks_v10gsm_20261101.tsv')
open(p, 'w').write('\n'.join(out) + '\n')
print(p, len(out))
