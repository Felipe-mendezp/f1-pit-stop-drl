"""Tabla de medias de posicion final con IC 95% (Figs. 6 y 7 v8) — 2026-10-20.

Lee fig6_2024_realdnf/*.csv y fig7_2024/*.csv (500 carreras por GP y metodo; la carrera i usa la semilla base+i en
todos los metodos, por lo que las diferencias con la QMIP se calculan pareadas). IC 95% normales: por GP,
media +- 1.96 * error estandar; promedio de los 8 GP con varianza sum(var_g / n) / 8^2. Solo ruido de simulacion
(un agente elegido por GP y algoritmo; no incluye la variabilidad entre semillas de entrenamiento).
    python results/world2024_v8_20261019/ci_table_20261020.py   # -> ci_table_20261020.tex
"""
import os, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
HEAD = ['Double DQN', 'A2C', 'TRPO', 'PPO', 'Rec. PPO', 'QMIP']
PANELS = [('fig6_2024_realdnf', 'Fully simulated races (Figure~\\ref{fig:stats_drivers_stochastic_2024})'),
          ('fig7_2024', 'Real race realizations (Figure~\\ref{fig:boxplot_drivers_2024})')]
Z = 1.96


def load(fig):
    d = pd.concat([pd.read_csv(f).assign(GP=os.path.basename(f)[:-4]) for f in sorted(glob.glob(f'{HERE}/{fig}/*.csv'))])
    d['i'] = d.groupby(['GP', 'algo']).cumcount()
    return d.pivot_table(index=['GP', 'i'], columns='algo', values='final_position')


def avg_ci(x):  # x: Series indexed by (GP, i)
    g = x.groupby('GP')
    return g.mean().mean(), Z * np.sqrt((g.var() / g.count()).sum()) / g.ngroups


lines = ['\\begin{tabular}{l' + 'c' * len(ALGOS) + '}', '\\toprule', 'GP & ' + ' & '.join(HEAD) + ' \\\\']
for fig, title in PANELS:
    P = load(fig)
    lines += ['\\midrule', f'\\multicolumn{{{len(ALGOS) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\', '\\midrule']
    for gp in P.index.get_level_values(0).unique():
        x = P.loc[gp]
        cells = [f'{x[a].mean():.2f} $\\pm$ {Z * x[a].sem():.2f}' for a in ALGOS]
        lines.append(gp.rsplit('_', 1)[0].replace('_', ' ') + ' & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    lines.append('Average & ' + ' & '.join('%.2f $\\pm$ %.2f' % avg_ci(P[a]) for a in ALGOS) + ' \\\\')
    lines.append('Difference with QMIP & ' + ' & '.join(
        '%+.2f $\\pm$ %.2f' % avg_ci(P[a] - P['QMIP']) for a in ALGOS[:-1]) + ' & -- \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
open(f'{HERE}/ci_table_20261020.tex', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
