"""2026-10-27 — Fig. 7 (dsq0) con la estrategia real 2024 simulada como referencia (caja gris "Actual", a la izquierda,
separada de los metodos por una linea vertical tenue). Mismos datos que figs67_2x4_20261023.py (F1_V9R2_DIR = dsq0) mas
results/real_strategy_check_20261027/fig7/ (mismas semillas). Salida <dsq0>/fig7_2024/figure_7_2x4_actual.{png,pdf}
    F1_V9R2_DIR=results/world2024_v9r2_20261023/dsq0 python scripts/gp_drivers_v5hd_20261014.py \
        results/world2024_v9r2_20261023/fig7_actual_20261027.py"""
import os, sys
from types import SimpleNamespace
HERE = os.environ.get('F1_V9R2_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dsq0')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()  # noqa: E401
import pandas as pd
import matplotlib.pyplot as plt
import evaluate_qmip_vs_drl as E6
VAL = os.path.join(os.environ.get('F1_REALSTRAT_DIR') or os.path.join(ROOT, 'results', 'real_strategy_check_20261027'), 'fig7')  # 2026-10-29: F1_REALSTRAT_DIR (v10eGQf)
E6.ALGO_ORDER = ['Actual'] + [a for a in E6.ALGO_ORDER if a != 'Actual']
E6.ALGO_COLORS['Actual'] = '#BDBDBD'
E6.ALGO_ABBREV['Actual'] = 'Actual'
_save = plt.savefig
def savefig(*a, **kw):   # linea vertical tenue entre la referencia y los metodos
    for ax in plt.gcf().axes:
        ax.axvline(1.5, color='0.6', lw=0.8, ls='-', zorder=0)
    return _save(*a, **kw)
plt.savefig = savefig
dd = os.path.join(HERE, 'fig7_2024')
data = []
for e in [e for e in E6.GP_CONFIGS if 'Emilia' not in e['gp_name']]:
    df = pd.read_csv(os.path.join(dd, e['model_dir'] + '.csv'))
    res = {a: [SimpleNamespace(final_position=int(p)) for p in g['final_position']] for a, g in df.groupby('algo')}
    v = pd.read_csv(os.path.join(VAL, e['model_dir'] + '.csv'))
    v = v[v.agent].sort_values('sim')
    assert len(v) == len(df[df.algo == 'PPO']), e['model_dir']
    res['Actual'] = [SimpleNamespace(final_position=int(p)) for p in v.position]
    start = E6.load_gp_config(e['gp_name'])['initial_positions_2024'].get(e['agent_driver'])
    data.append(dict(display_name=df['display_name'].iloc[0], results=res,
                     real_starting_pos=start, real_finishing_pos=e['real_finishing_pos']))
for ext in ('png', 'pdf'):
    E6.plot_grid_boxplot(data, save_path=os.path.join(dd, f'figure_7_2x4_actual.{ext}'), nrows=2, ncols=4,
                         figsize=(22, 8), legend_space=0.08)
