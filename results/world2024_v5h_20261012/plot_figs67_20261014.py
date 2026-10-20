"""Figs. 6 y 7 v5h (mundo 2024) desde los CSV de esta carpeta — 2026-10-14 (preliminar, no es la figura del paper).

Usa plot_grid_boxplot de scripts/evaluate_qmip_vs_drl.py (misma funcion que la Fig. 6 publicada; la de la Fig. 7 es
identica salvo el formato). Lineas de referencia (posicion real de partida y de llegada 2024) desde GP_CONFIGS.
Orden de paneles = GP_CONFIGS sin Emilia Romagna (8 GP).

    python results/world2024_v5h_20261012/plot_figs67_20261014.py   ->  fig6_2024/figure_6.{png,pdf}, fig7_2024/figure_7.{png,pdf}
"""
import os, sys
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()
import pandas as pd
import evaluate_qmip_vs_drl as E6

for fig in (6, 7):
    d = os.path.join(HERE, f'fig{fig}_2024')
    data = []
    for entry in E6.GP_CONFIGS:
        csv = os.path.join(d, f"{entry['model_dir']}.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        res = {a: [SimpleNamespace(final_position=int(p)) for p in g['final_position']] for a, g in df.groupby('algo')}
        start = E6.load_gp_config(entry['gp_name'])['initial_positions_2024'].get(entry['agent_driver'])
        data.append(dict(display_name=df['display_name'].iloc[0], results=res,
                         real_starting_pos=start, real_finishing_pos=entry['real_finishing_pos']))
    assert len(data) == 8, len(data)
    for ext in ('png', 'pdf'):
        E6.plot_grid_boxplot(data, save_path=os.path.join(d, f'figure_{fig}.{ext}'))
