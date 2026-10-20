"""Resumen de las Figs. 6/7 v6 (mundo 2024) y del chequeo de paradas tardias — 2026-10-14.

Lee <D>/fig6_2024_realdnf, <D>/fig7_2024 y <D>/dnfobs/fig7_2024 (D = --dir, por defecto esta carpeta) y, por GP y
metodo: posicion media, paradas reales medias (n_pits_all) y % de carreras con alguna parada en las ultimas 6 vueltas.
Para la Fig. 7 agrega la referencia v5h (results/world2024_v5h_pits_20261014/fig7_2024, GP sin cambio de piloto) y la
intervencion dnfobs (primeras 200 carreras, mismas semillas): si v6 aprendio a ignorar al rival retirado, dnfobs ~ base.
Si estan los 8 GP de una figura, dibuja figure_<6|7>.{png,pdf} (misma funcion que la Fig. 6 publicada).
Correr via scripts/gp_drivers_v5hd_20261014.py (nombres de carpeta y pilotos v6).
"""
import os, sys, argparse
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()  # noqa: E401
import numpy as np
import pandas as pd
import evaluate_qmip_vs_drl as E6
from model_loader import ModelLoader

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=HERE)
ap.add_argument('--no-plot', action='store_true')
args = ap.parse_args()
D = args.dir
V5H = os.path.join(ROOT, 'results', 'world2024_v5h_pits_20261014', 'fig7_2024')
A = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
ENTRIES = [e for e in E6.GP_CONFIGS if 'Emilia' not in e['gp_name']]
loader = ModelLoader()
NL = {e['model_dir']: loader.load(e['gp_name']).n_laps for e in ENTRIES}


def stats(df, n, head=None):
    out = {}
    for a, x in df.groupby('algo'):
        x = x.head(head) if head else x
        laps = x.pit_laps.fillna('').astype(str).apply(lambda s: [int(float(v)) for v in s.split()])
        out[a] = dict(pos=x.final_position.mean(), pits=x.n_pits_all.mean(),
                      tard=100 * np.mean([any(l > n - 6 for l in ls) for ls in laps]))
    return out


rows = []
for e in ENTRIES:
    g, n = e['model_dir'], NL[e['model_dir']]
    srcs = [('fig6_realdnf', os.path.join(D, 'fig6_2024_realdnf', g + '.csv'), None),
            ('fig7', os.path.join(D, 'fig7_2024', g + '.csv'), None),
            ('fig7_base200', os.path.join(D, 'fig7_2024', g + '.csv'), 200),
            ('fig7_dnfobs200', os.path.join(D, 'dnfobs', 'fig7_2024', g + '.csv'), 200),
            ('fig7_v5h', os.path.join(V5H, g + '.csv'), None)]
    for name, p, head in srcs:
        if os.path.isfile(p):
            for a, s in stats(pd.read_csv(p), n, head).items():
                rows.append(dict(GP=g, src=name, algo=a, **s))
if not rows:
    sys.exit(f'sin resultados en {D}')
d = pd.DataFrame(rows)
d.to_csv(os.path.join(D, 'summary_v6_20261014.csv'), index=False)
pd.set_option('display.width', 220)
for c, t in [('pos', 'posicion media'), ('pits', 'paradas reales medias'),
             ('tard', '% carreras con parada en las ultimas 6 vueltas')]:
    print(f'\n=== {t}')
    pt = d.pivot_table(index=['GP', 'src'], columns='algo', values=c, sort=False)
    print(pt[[a for a in A if a in pt.columns]].round(2).to_string())

if not args.no_plot:
    for fig, sub in ((6, 'fig6_2024_realdnf'), (7, 'fig7_2024')):
        dd = os.path.join(D, sub)
        data = []
        for e in ENTRIES:
            csv = os.path.join(dd, e['model_dir'] + '.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            res = {a: [SimpleNamespace(final_position=int(p)) for p in g['final_position']] for a, g in df.groupby('algo')}
            start = E6.load_gp_config(e['gp_name'])['initial_positions_2024'].get(e['agent_driver'])
            data.append(dict(display_name=df['display_name'].iloc[0], results=res,
                             real_starting_pos=start, real_finishing_pos=e['real_finishing_pos']))
        if len(data) == len(ENTRIES):
            for ext in ('png', 'pdf'):
                E6.plot_grid_boxplot(data, save_path=os.path.join(dd, f'figure_{fig}.{ext}'))
            print(f'figura {fig}: {dd}/figure_{fig}.png')
        else:
            print(f'figura {fig}: {len(data)}/{len(ENTRIES)} GP, no se dibuja')
