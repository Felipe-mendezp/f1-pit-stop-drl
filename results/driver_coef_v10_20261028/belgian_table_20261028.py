"""2026-10-28 — tab:regression_pitstops (supl.): Ec. 2 porcentual estimada con las carreras previas al GP de Belgica 2024,
especificacion completa y sin X_Q (la usada, por la clasificacion con lluvia). Para control/ (debe reproducir la tabla
del paper: 929/1022 obs., R2 0.581/0.519), v10/, v10c/ y v10d/ Correr desde la raiz del repo.
Salida: belgian_table_20261028.csv
"""
import os, importlib.util
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for v in ('control', 'v10', 'v10c', 'v10d', 'v10eGQ'):   # 28-oct: + v10c y v10d; 28-oct noche: + v10eGQ
    s = importlib.util.spec_from_file_location('eq2_' + v, os.path.join(HERE, v, 'eq2_fallbacks_20261014.py'))
    M = importlib.util.module_from_spec(s); s.loader.exec_module(M)
    E = M.Eq2()
    t = 2024 * 100 + int(E.rnd[(2024, 'Belgian Grand Prix')])
    for name, q in (('completa', True), ('sin_Q', False)):
        m = E.model(t, M.spec(3, True, q))
        for k in m.params.index:
            rows.append(dict(version=v, spec=name, var=k, coef=m.params[k], se=m.bse[k], p=m.pvalues[k],
                             nobs=int(m.nobs), r2=m.rsquared))
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, 'belgian_table_20261028.csv'), index=False)
pd.set_option('display.width', 200)
print(T.pivot_table(index=['spec', 'var'], columns='version', values=['coef', 'se']).round(4).to_string())
print(T.groupby(['version', 'spec'])[['nobs', 'r2']].first().round(3).to_string())
