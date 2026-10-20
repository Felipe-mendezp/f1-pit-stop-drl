"""2026-10-09 — Bootstrap por GP (pareado) de la diferencia de error de validación 2024 entre calibraciones.
Lee pooled_20261004/validation_cells.csv. Uso: python bootstrap_options_20261009.py"""
import os, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(D, 'pooled_20261004', 'validation_cells.csv'))
v['age'] = v.age.fillna(-1)
PAPER = ['Bahrain', 'Saudi', 'Miami', 'Hungarian', 'Belgian', 'Dutch', 'Singapore', 'United States']
v['paper'] = v.GP.apply(lambda g: any(g.startswith(p) for p in PAPER))
key = ['GP', 'kind', 'pair', 'age']
w = v.pivot_table(index=key + ['se', 'paper'], columns='opcion', values='err').reset_index()
rng = np.random.default_rng(2026); B = 10000
rows = []
for conj, sub in [('8 GP paper', w[w.paper]), ('todos', w)]:
    for kind in ['delta', 'slope']:
        s = sub[sub.kind == kind]
        for a, b in [('opcion2', 'opcion3'), ('actual', 'opcion2'), ('actual', 'opcion3')]:
            c = s.dropna(subset=[a, b]).copy()
            c['d'] = c[a].abs() - c[b].abs()
            c['dw'] = c['d'] / c.se ** 2; c['w'] = 1 / c.se ** 2
            g = c.groupby('GP').agg(d=('d', 'sum'), n=('d', 'size'), dw=('dw', 'sum'), w=('w', 'sum'))
            gps = g.index.values; est = g.d.sum() / g.n.sum(); estw = g.dw.sum() / g.w.sum()
            bs, bsw = [], []
            for _ in range(B):
                k = g.loc[rng.choice(gps, len(gps))]
                bs.append(k.d.sum() / k.n.sum()); bsw.append(k.dw.sum() / k.w.sum())
            bs, bsw = np.array(bs), np.array(bsw)
            gw = (g.d / g.n > 0).sum()
            rows.append(dict(conjunto=conj, tipo=kind, comparacion=f'{a} - {b}', n_GP=len(gps), n_celdas=len(c),
                             dif_MAE=est, IC95_lo=np.quantile(bs, .025), IC95_hi=np.quantile(bs, .975),
                             p_boot=2 * min((bs <= 0).mean(), (bs >= 0).mean()),
                             dif_MAE_w=estw, IC95w_lo=np.quantile(bsw, .025), IC95w_hi=np.quantile(bsw, .975),
                             GP_donde_a_peor=f'{gw}/{len(gps)}'))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(D, 'bootstrap_options_20261009.csv'), index=False)
pd.set_option('display.width', 250)
print(r.round(4).to_string(index=False))
c = w[w.paper & (w.kind == 'delta')].dropna(subset=['opcion2', 'opcion3'])
print('\nMAE delta por GP (8 GP paper):')
print(c.groupby('GP')[['actual', 'opcion2', 'opcion3']].agg(lambda x: x.abs().mean()).round(3).to_string())
