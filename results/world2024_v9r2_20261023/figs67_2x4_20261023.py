"""2026-10-23 — Figs. 6 and 7 v9r2 in the 2x4 layout of the paper. Copy of
results/world2024_v8_qmipsym_20261021/figs67_2x4_20261021.py reading fig6_2024_realdnf/ and fig7_2024/ of this folder
(no merge: the v9r2 runs already have the QMIP with sets). Output <fig dir>/figure_<6|7>_2x4.*
    python scripts/gp_drivers_v5hd_20261014.py results/world2024_v9r2_20261023/figs67_2x4_20261023.py"""
import os, sys
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.environ.get('F1_V9R2_DIR') or HERE   # 2026-10-23: otra carpeta de datos (pruebas)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()  # noqa: E401
import pandas as pd
import evaluate_qmip_vs_drl as E6
for fig, sub in ((6, 'fig6_2024_realdnf'), (7, 'fig7_2024')):
  dd = os.path.join(HERE, sub)
  data = []
  for e in [e for e in E6.GP_CONFIGS if 'Emilia' not in e['gp_name']]:
      df = pd.read_csv(os.path.join(dd, e['model_dir'] + '.csv'))
      res = {a: [SimpleNamespace(final_position=int(p)) for p in g['final_position']] for a, g in df.groupby('algo')}
      start = E6.load_gp_config(e['gp_name'])['initial_positions_2024'].get(e['agent_driver'])
      data.append(dict(display_name=df['display_name'].iloc[0], results=res,
                       real_starting_pos=start, real_finishing_pos=e['real_finishing_pos']))
  for ext in ('png', 'pdf'):
      E6.plot_grid_boxplot(data, save_path=os.path.join(dd, f'figure_{fig}_2x4.{ext}'), nrows=2, ncols=4, figsize=(22, 8), legend_space=0.08)
