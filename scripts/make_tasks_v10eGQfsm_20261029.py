"""2026-10-29 — Tareas del post-entrenamiento con largada sorteada v10eGQ con el adelantamiento v10f (App. Q; copia de
scripts/make_tasks_v10sm_20261027.py, que queda para v10eGQ sin v10f).

Cambios respecto de make_tasks_v10sm_20261027.py: agentes de partida trained_models/retrained_v10eGQf_20261029/selected
(solo lectura; hay que subirlos al cluster sin --delete), salida trained_models/retrained_v10eGQf_startmix_20261029/,
scripts/overtake_v10f_20261029.py primero en el envoltorio y --following-gap-scale 0.26 (gap de seguimiento
0.4 + Exp(0.26); results/sim_v10eGQ_20261028/README_adelantamiento_20261029.md). Resto igual: 8 GP x {PPO, RecurrentPPO}
x 3 semillas = 48 tareas, p = 0.5, 1.5 M pasos, lr constante PPO 1e-4 / RPPO 1.5e-4, arbol simulation_v10eGQ_20261028.
Correr con los envoltorios (GP_CONFIGS con los pilotos v5hd):
    python scripts/tyre_ages_v10eGQ_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py \
        scripts/rival_start_v8_20261018.py scripts/make_tasks_v10eGQfsm_20261029.py  ->  reproduce/cluster/tasks_v10eGQfsm_20261029.tsv
"""
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_all_gps import GP_CONFIGS

SEL = "trained_models/retrained_v10eGQf_20261029/selected"
OUT = 'trained_models/retrained_v10eGQf_startmix_20261029'
DST = 'reproduce/cluster/tasks_v10eGQfsm_20261029.tsv'
WRAP = 'scripts/overtake_v10f_20261029.py scripts/tyre_ages_v10eGQ_20261028.py scripts/v10_wrapper_20261027.py scripts/tyre_reuse_v9r2_20261022.py scripts/tyre_sets_v9_20261021.py scripts/rival_start_v8_20261018.py'
LR = {'PPO': '1e-4', 'RecurrentPPO': '1.5e-4'}
assert os.path.isdir(SEL), f'falta {SEL} (correr la seleccion del multi-piloto v10eGQf)'
rows = []
for e in [e for e in GP_CONFIGS if os.path.isdir(os.path.join(SEL, e["model_dir"]))]:
    for seed in (1, 2, 3):
        for algo in ('RecurrentPPO', 'PPO'):
            name = f"v10eGQfsm__seed{seed:02d}__{e['model_dir']}__{algo}"
            cmd = (f"{WRAP} scripts/finetune_startmix_20261025.py --agents {SEL} "
                   f"--gp '{e['gp_name']}' --algo {algo} --lr {LR[algo]} --start-mix 0.5 --seed {seed} "
                   f"--timesteps 1500000 --eval-freq 5000 --vec dummy --fast-env --torch-threads 4 --following-gap-scale 0.26 "
                   f"--sc-bunching --sc-bunch-delta 1.5 --out-root {OUT}/seed{seed:02d}")
            rows.append('\t'.join([name, f'{OUT}/logs/{name}.log', 'trained_models/simulation_v10eGQ_20261028', cmd]))
with open(DST, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(len(rows), 'tareas ->', DST)
