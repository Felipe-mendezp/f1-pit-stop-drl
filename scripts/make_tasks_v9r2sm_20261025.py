"""2026-10-25 — Tareas del post-entrenamiento con largada sorteada (v9r2sm; docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md).

8 GP x {PPO, RecurrentPPO} x 3 semillas de fine-tuning = 48 tareas, desde los agentes v9r2 seleccionados
(trained_models/rl_agents_v9r2_20261022/selected, solo lectura) hacia trained_models/retrained_v9r2_startmix_20261025/.
p = 0.5, 1.5 M pasos, lr constante PPO 1e-4 / RPPO 1.5e-4, misma fisica y envoltorio que tasks_v9r2.tsv.
Correr con el envoltorio v9r2 (GP_CONFIGS con los pilotos v5hd, como los agentes seleccionados):
    python scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py \
        scripts/make_tasks_v9r2sm_20261025.py  ->  reproduce/cluster/tasks_v9r2sm.tsv
"""
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_all_gps import GP_CONFIGS

SEL = "trained_models/rl_agents_v9r2_20261022/selected"

OUT = 'trained_models/retrained_v9r2_startmix_20261025'
WRAP = 'scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py'
LR = {'PPO': '1e-4', 'RecurrentPPO': '1.5e-4'}
rows = []
for e in [e for e in GP_CONFIGS if os.path.isdir(os.path.join(SEL, e["model_dir"]))]:
    for seed in (1, 2, 3):
        for algo in ('RecurrentPPO', 'PPO'):
            name = f"v9r2sm__seed{seed:02d}__{e['model_dir']}__{algo}"
            cmd = (f"{WRAP} scripts/finetune_startmix_20261025.py --agents trained_models/rl_agents_v9r2_20261022/selected "
                   f"--gp '{e['gp_name']}' --algo {algo} --lr {LR[algo]} --start-mix 0.5 --seed {seed} "
                   f"--timesteps 1500000 --eval-freq 5000 --fast-env --torch-threads 4 --following-gap-scale 1.0 "
                   f"--sc-bunching --sc-bunch-delta 1.5 --out-root {OUT}/seed{seed:02d}")
            rows.append('\t'.join([name, f'{OUT}/logs/{name}.log', 'trained_models/simulation_v7_20261015', cmd]))
with open('reproduce/cluster/tasks_v9r2sm.tsv', 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(len(rows), 'tareas -> reproduce/cluster/tasks_v9r2sm.tsv')
