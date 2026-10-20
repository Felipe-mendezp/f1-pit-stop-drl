"""2026-10-11 — Opcion 3 (y 2) con covarianza de la Ec. 1 agrupada por stint.

Igual que pooled_20261004.py, salvo fit_gp: la covarianza de los coeficientes de compuesto de cada GP es
robusta agrupada por stint (anio x piloto x stint) en vez de la OLS clasica. Las vueltas de un stint no son
independientes, asi que la OLS sobrestima la precision; el GLS del modelo conjunto pondera cada GP por esa
precision. Motivo: Saudi (corr(TyreLife, LapNumber) = 0.82 en 2021-23) sale con degradacion ~0.

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/pooled_cluster_20261011.py
Salida: results/tyre_coef_study_20260930/pooled_cluster_20261011/ (mismos archivos que pooled_20261004/).
"""
import io, os, sys, contextlib, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('pooled', os.path.join(HERE, 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
R = P.R


def fit_gp_cluster(d, gp):
    with contextlib.redirect_stdout(io.StringIO()):
        m, dd = R.filtrar_outliers_iqr(d, gp)
    rows = m.model.data.row_labels
    grp = pd.factorize(dd.loc[rows, ['Year', 'Driver', 'stint']].astype(str).agg('|'.join, axis=1))[0]
    mc = m.get_robustcov_results(cov_type='cluster', groups=grp)
    names_all = m.model.exog_names
    Pp = pd.Series(mc.params, index=names_all); V = pd.DataFrame(mc.cov_params(), index=names_all, columns=names_all)
    ref = Pp.index[Pp.index.str.contains('Compound_Detail, Treatment')][0].split("'")[1]
    comps = sorted(dd.Compound_Detail.unique())
    names, kinds = [], []
    for c in comps:
        if c != ref:
            names.append(f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"); kinds.append(('off', c))
    for c in comps:
        names.append(f"TyreLife:C(Compound_Detail)[{c}]"); kinds.append(('sl', c))
    stints = {c: s.groupby(['Year', 'Driver', 'stint']).ngroups for c, s in dd.groupby('Compound_Detail')}
    return dict(ref=ref, comps=comps, kinds=kinds, b=Pp[names].values, V=V.loc[names, names].values,
                laps=dd.Compound_Detail.value_counts().to_dict(), stints=stints,
                se_ols=np.sqrt(np.diag(m.cov_params().loc[names, names].values)))


P.fit_gp = fit_gp_cluster
P.OUT = os.path.join(HERE, 'pooled_cluster_20261011')
os.makedirs(P.OUT, exist_ok=True)

if __name__ == '__main__':
    P.main()
