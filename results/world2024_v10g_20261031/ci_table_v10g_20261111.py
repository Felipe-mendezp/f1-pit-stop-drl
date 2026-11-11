r"""Tabla P.1 v10g con marcas de significancia (2026-11-11): copia de results/world2024_v9r2_20261023/ci_table_v9r2_20261023.py
con el mismo calculo de medias e IC 95 %, mas una marca por celda: $^{*}$ si el metodo es mejor que la QMIP (menor
posicion) y $^{\dagger}$ si es peor (11-nov: antes +/-), al 5 % con prueba t pareada de dos colas (carrera i = misma semilla en todos los metodos),
es decir, cuando el IC 95 % de la diferencia pareada excluye el cero. Sin correccion por comparaciones multiples
(decision del autor, 11-nov). Fila Average: misma regla con el IC del promedio de los 8 GP (varianza sum(var_g/n)/8^2).
Las marcas van en \\vten{} para el manuscrito.
    python results/world2024_v10g_20261031/ci_table_v10g_20261111.py   # -> ci_table_v10g_20261111.tex
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALGOS = ['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO', 'QMIP']
HEAD = ['Double DQN', 'A2C', 'TRPO', 'PPO', 'Rec. PPO', 'QMIP']
PANELS = [('fig6_2024_realdnf', 'Fully simulated races (Figure~\\ref{fig:stats_drivers_stochastic_2024})'),
          ('fig7_2024', 'Real race realizations (Figure~\\ref{fig:boxplot_drivers_2024})')]
Z = 1.96
ALPHA = 0.05


def load(fig):
    d = pd.concat([pd.read_csv(f).assign(GP=os.path.basename(f)[:-4]) for f in sorted(glob.glob(f'{HERE}/{fig}/*.csv'))])
    d['i'] = d.groupby(['GP', 'algo']).cumcount()
    return d.pivot_table(index=['GP', 'i'], columns='algo', values='final_position')


def avg_ci(x):  # x: Series indexed by (GP, i)
    g = x.groupby('GP')
    return g.mean().mean(), Z * np.sqrt((g.var() / g.count()).sum()) / g.ngroups


def mark(mean, p):
    if p >= ALPHA or mean == 0:
        return ''
    return '\\vten{$^{*}$}' if mean < 0 else '\\vten{$^{\\dagger}$}'


def mark_gp(x, a):
    if a == 'QMIP':
        return ''
    d = (x[a] - x['QMIP']).to_numpy()
    if np.all(d == d[0]):
        return ''
    return mark(d.mean(), stats.ttest_rel(x[a], x['QMIP']).pvalue)


def mark_avg(P, a):
    if a == 'QMIP':
        return ''
    m, ci = avg_ci(P[a] - P['QMIP'])
    se = ci / Z
    return mark(m, 2 * stats.norm.sf(abs(m / se)))


lines = ['\\begin{tabular}{l' + 'c' * len(ALGOS) + '}', '\\toprule', 'GP & ' + ' & '.join(HEAD) + ' \\\\']
log = []
for fig, title in PANELS:
    P = load(fig)
    lines += ['\\midrule', f'\\multicolumn{{{len(ALGOS) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\', '\\midrule']
    for gp in P.index.get_level_values(0).unique():
        x = P.loc[gp]
        cells = [f'{x[a].mean():.2f} $\\pm$ {Z * x[a].sem():.2f}{mark_gp(x, a)}' for a in ALGOS]
        lines.append(gp.rsplit('_', 1)[0].replace('_', ' ') + ' & ' + ' & '.join(cells) + ' \\\\')
        for a in ALGOS[:-1]:
            log.append((fig, gp, a, (x[a] - x['QMIP']).mean(), stats.ttest_rel(x[a], x['QMIP']).pvalue, mark_gp(x, a)))
    lines.append('\\midrule')
    lines.append('Average & ' + ' & '.join('%.2f $\\pm$ %.2f' % avg_ci(P[a]) + mark_avg(P, a) for a in ALGOS) + ' \\\\')
    lines.append('Difference with QMIP & ' + ' & '.join(
        '%+.2f $\\pm$ %.2f' % avg_ci(P[a] - P['QMIP']) for a in ALGOS[:-1]) + ' & -- \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
open(f'{HERE}/ci_table_v10g_20261111.tex', 'w').write('\n'.join(lines) + '\n')
pd.DataFrame(log, columns=['fig', 'gp', 'algo', 'mean_diff', 'p_ttest_rel', 'mark']).to_csv(
    f'{HERE}/ci_table_v10g_20261111_tests.csv', index=False)
print('\n'.join(lines))
