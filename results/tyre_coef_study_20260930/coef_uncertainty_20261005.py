"""2026-10-05 — Incertidumbre de los coeficientes de neumatico del simulador PUBLICADO (8 GP del paper).

Por GP y compuesto usado en 2024 (config.json del simulador), offset y pendiente que usa trained_models/simulation/<GP>/laptime_clear.pkl, y si el
valor es la estimacion de la Ec. 1 (data/compounds_coefs_V3.xlsx) o fue imputado por la completacion.
  estimado : sigma = error estandar agrupado por stint (misma regresion e IQR k=3 que el pipeline)
  imputado : pendiente -> CV medio (se/coef) de las pendientes estimadas del GP x valor imputado
             offset    -> se medio de los offsets estimados del GP (sin la referencia)
             (factor de amplificacion = 1; si el GP no tiene estimados de ese tipo, promedio de los 8 GP)
Solo lectura. Correr desde la raiz del repo:
    python results/tyre_coef_study_20260930/coef_uncertainty_20261005.py
Salida: results/tyre_coef_study_20260930/coef_uncertainty_20261005.csv
"""
import contextlib, io, json, os, pickle, sys
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import models.regressions as R

GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coef_uncertainty_20261005.csv')
v3 = pd.read_excel(os.path.join(ROOT, 'data', 'compounds_coefs_V3.xlsx'))
laps = R.all_laps()
laps['GP'] = laps['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')


def clustered_fit(gp):
    with contextlib.redirect_stdout(io.StringIO()):
        mod, d = R.filtrar_outliers_iqr(laps, gp)
    d = d.copy()
    d['Year'] = d.Year.astype(int).astype(str)
    d = d.sort_values(['Year', 'Driver', 'LapNumber']).reset_index(drop=True)
    g = d.groupby(['Year', 'Driver'])
    new = (g.TyreLife.diff() <= 0) | (g.Compound_Detail.shift() != d.Compound_Detail)
    cl = d.Year + d.Driver + new.groupby([d.Year, d.Driver]).cumsum().astype(str)
    return smf.ols(mod.model.formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(cl)[0]})


rows = []
for gp in GPS:
    sim = pickle.load(open(os.path.join(ROOT, 'trained_models', 'simulation', gp, 'laptime_clear.pkl'), 'rb'))
    s = pd.Series(sim.coef_, index=sim.feature_names_)
    ref = sim.reference_categories_['Compound_Detail']
    est = v3[v3.GP == gp].set_index('Variable')
    m = clustered_fit(gp)
    assert np.allclose(m.params[est.index], est.Coef, atol=1e-6), gp  # misma regresion que el pipeline
    comps = json.load(open(os.path.join(ROOT, 'trained_models', 'simulation', gp, 'config.json')))['compounds']
    for c in comps:
        for kind, name in [('offset', f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"),
                           ('pendiente', f'TyreLife:C(Compound_Detail)[{c}]')]:
            if kind == 'offset' and c == ref:
                rows.append(dict(GP=gp, ref=ref, compuesto=c, tipo=kind, valor=0.0, estado='referencia'))
                continue
            val = float(s[name])
            e = est.Coef.get(name)
            kept = e is not None and abs(e - val) < 1e-6
            rows.append(dict(GP=gp, ref=ref, compuesto=c, tipo=kind, valor=val,
                             estado='estimado' if kept else 'imputado',
                             estimado_ec1=e, p_ols=est['p-value'].get(name),
                             se_cluster=float(m.bse[name]) if name in m.bse else np.nan,
                             p_cluster=float(m.pvalues[name]) if name in m.pvalues else np.nan))
t = pd.DataFrame(rows)

# sigma: estimados -> se agrupado; imputados -> regla del GP (o promedio de los 8 GP)
est_rows = t[t.estado == 'estimado']
cv_gp = (est_rows[est_rows.tipo == 'pendiente'].assign(cv=lambda x: x.se_cluster / x.valor)
         .groupby('GP').cv.mean())
se_off_gp = est_rows[est_rows.tipo == 'offset'].groupby('GP').se_cluster.mean()
cv_all, se_off_all = cv_gp.mean(), se_off_gp.mean()
def sigma(r):
    if r.estado == 'estimado':
        return r.se_cluster
    if r.estado == 'referencia':
        return 0.0
    if r.tipo == 'pendiente':
        return cv_gp.get(r.GP, cv_all) * r.valor
    return se_off_gp.get(r.GP, se_off_all)
t['sigma_f1'] = t.apply(sigma, axis=1)
t['regla'] = [('' if r.estado != 'imputado' else
               ('CV GP' if (r.tipo == 'pendiente' and r.GP in cv_gp) or (r.tipo == 'offset' and r.GP in se_off_gp)
                else 'promedio 8 GP')) for r in t.itertuples()]
t.to_csv(OUT, index=False)
pd.set_option('display.width', 200)
print(t[['GP', 'compuesto', 'tipo', 'valor', 'estado', 'estimado_ec1', 'p_ols', 'p_cluster', 'sigma_f1', 'regla']]
      .round(4).to_string(index=False))
print(f'\nCV medio pendientes estimadas (8 GP): {cv_all:.3f} | se medio offsets estimados: {se_off_all:.3f}')
print('->', OUT)
