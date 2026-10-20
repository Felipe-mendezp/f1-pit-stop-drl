"""2026-10-11 — Resumen del diagnostico Hungarian v5 (scripts/diag_hungarian_v5_20261011.py).
Por mundo (rivales x amarillas) y politica: posicion media, diferencia pareada con la QMIP (misma carrera i:
misma semilla y mismo escenario SC/VSC), paradas y planes mas frecuentes. En mundos con amarillas
estocasticas, separa carreras con y sin SC/VSC. Escribe summary_20261011.csv."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, 'r*__*.csv')) if '_trace' not in f], ignore_index=True)
d['plan'] = d.start.astype(str) + '|' + d.stops.fillna('')
C = {3: 'H', 4: 'M', 5: 'S'}
d['seq'] = d.start.map(C) + d.stops.fillna('').str.split().apply(lambda s: ''.join(C[int(x.split(':')[1])] for x in s))
rows = []
for (r, y), g in d.groupby(['rivals', 'yf']):
    q = g[g.policy == 'QMIP'].set_index('sim')
    yfmask = q.any_yf.astype(bool)
    for pol, h in g.groupby('policy'):
        h = h.set_index('sim').loc[q.index]
        dd = q.final_position - h.final_position          # > 0: la politica gana a la QMIP
        row = dict(rivals=r, yf=y, policy=pol, n=len(h), pos=h.final_position.mean(),
                   gap_vs_qmip=dd.mean(), ci=1.96 * dd.std() / np.sqrt(len(dd)),
                   stops_mean=h.n_stops.mean(), top_seq=h.seq.value_counts(normalize=True).round(2).head(3).to_dict())
        if y == 'stoch':
            row.update(pos_conYF=h.final_position[yfmask].mean(), pos_sinYF=h.final_position[~yfmask].mean(),
                       gap_conYF=dd[yfmask].mean(), gap_sinYF=dd[~yfmask].mean(), share_YF=yfmask.mean())
        rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(HERE, 'summary_20261011.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(R.round(2).to_string(index=False))
