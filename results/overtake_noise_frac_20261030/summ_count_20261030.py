"""2026-10-30 — Resumen de ot_count/*.pkl, en el formato de overtake_diag_20261029/sim_variantes.txt (fila real copiada
de ahi: 21 carreras reales 2021-2024 de los 8 GP)."""
import os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
band = lambda g: pd.cut(g, [-np.inf, 1, 2, np.inf], labels=['<=1s', '1-2s', '>2s'])
rows = [dict(var='real', le1=0.196, p10=0.652, p50=2.690, **{'<=1s': 288.762, '1-2s': 67.81, '>2s': 113.524},
             rate_le1=0.229, rate_12=0.056)]
for f in sorted(glob.glob(os.path.join(HERE, 'ot_count', '*.pkl'))):
    q = pd.read_pickle(f); q['band'] = band(q.gap); qq = q[q.gap < 60]
    c = q.groupby('band', observed=False).ot.sum() / q.groupby(['GP', 'seed']).ngroups * 8
    rows.append(dict(var=os.path.basename(f)[:-4], le1=(qq.gap <= 1).mean(), p10=qq.gap.quantile(.1), p50=qq.gap.median(),
                     **c.to_dict(), rate_le1=q[q.band == '<=1s'].ot.mean(), rate_12=q[q.band == '1-2s'].ot.mean()))
X = pd.DataFrame(rows); X['total'] = X[['<=1s', '1-2s', '>2s']].sum(1)
print('adelantamientos: suma de los 8 GP de la media por carrera; gaps: pares adyacentes al inicio de vuelta verde, vueltas 5+')
print(X.round(3).to_string(index=False))
