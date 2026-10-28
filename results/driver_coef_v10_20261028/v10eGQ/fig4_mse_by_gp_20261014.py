"""2026-10-14 — Datos de la Fig. 4 (MSE por GP 2024 de los coeficientes de piloto, con y sin correccion).

Correr desde la raiz del repo: python results/driver_coef_study_20261010/fig4_mse_by_gp_20261014.py
Con correccion = prediccion de la Ec. 2 porcentual CON respaldos y la regla de v6 (eq2_fallbacks_20261014.Eq2.predict),
es decir, lo que usa el simulador v6 (p. ej. Belgian 2024 sin Q). Se omiten las carreras sin verdad definida
(piloto de referencia con < 15 vueltas limpias: Australia 2024). Sin correccion = coeficiente de la Ec. 1 estimada con las
ediciones 2021..2023 del mismo GP (columna hist de eq2_features.csv). Verdad = Ec. 1 con solo la carrera 2024.
Muestra por GP: pilotos != VER con >= 15 vueltas limpias y con ambas predicciones (misma muestra para las dos).
Salida: fig4_mse_by_gp_20261014.csv y fig4_tikz_coords_20261014.txt (coordenadas para el TikZ del paper).
"""
import os, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('eq2', os.path.join(HERE, 'eq2_fallbacks_20261014.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
E = M.Eq2()
S = E.sample[E.sample.Year == 2024]
rows = []
for r in S.itertuples():
    p = E.predict(2024, r.GP, r.Driver)
    rows.append(dict(GP=r.GP, Driver=r.Driver, truth=r.coef, corr=p['pred'], spec=p['spec'], hist=r.hist))
D = pd.DataFrame(rows).dropna(subset=['truth', 'corr', 'hist'])
G = D.groupby('GP').apply(lambda x: pd.Series(dict(n=len(x), with_corr=((x['corr'] - x.truth) ** 2).mean(),
                                                   without_corr=((x['hist'] - x.truth) ** 2).mean(),
                                                   spec_mode=x.spec.mode().iloc[0]))).reset_index()
G['Round'] = [int(E.rnd[(2024, g)]) for g in G.GP]
G = G.sort_values('Round')
G.to_csv(os.path.join(HERE, 'fig4_mse_by_gp_20261014.csv'), index=False)
print(G.round(3).to_string(index=False))
print('media con %.3f sin %.3f; GP donde la correccion empeora:' % (G.with_corr.mean(), G.without_corr.mean()),
      list(G[G.with_corr > G.without_corr].GP))
with open(os.path.join(HERE, 'fig4_tikz_coords_20261014.txt'), 'w') as f:
    f.write(',\n'.join(g.replace(' Grand Prix', '') for g in G.GP) + '\n\n')
    for c in ('with_corr', 'without_corr'):
        f.write(c + '\n' + '\n'.join(f'    ({i}, {v:.3f})' for i, v in enumerate(G[c], 1)) + '\n\n')
