"""2026-10-15: factor unico de TyreLife 2021, estimacion conjunta (decision del autor: un solo factor).

Dos estimaciones del mismo f, ambas con efectos propios de 2021 en la tendencia de la carrera:
  fisica:     Ec. 1 por GP con LapNumber y LapNumber^2 por anio (laptime_factor_20261015.py, robustez): 0.660, ee 0.072
              (agrupado por carrera, Gauss-Newton);
  conductual: logit de parada con curva de tramo de carrera propia de 2021 (stop_logit_factor_20261015.py, L3).
Aqui se recalcula el ee del conductual agrupado por carrera (anio x GP), para que ambos sean comparables: se linealiza
el indice en f (z = y21 * TyreLife * pendiente del compuesto, evaluada en f estimado) y se agrega z al logit; el ee
agrupado del coeficiente de z es el ee de f.
Conjunta: distancia minima (promedio ponderado por 1/ee^2), suponiendo estimaciones independientes (usan las mismas
carreras pero datos y variacion distintos: tiempos de vuelta frente a decisiones de parar). Se reporta la prueba de
compatibilidad (chi2 con 1 gl). Sensibilidades: conductual L4; fisica con curva comun (0.555) y conductual L2.
"""
import os, sys
os.environ['TLV'] = 'ninguno'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rival_model_eval_20261014'))
import rivals_common_20261014 as RC
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

train, _ = RC.load()
train['race'] = train.Year.astype(str) + '|' + train.GP
tl0 = train.TyreLife.values.copy()
lt = pd.read_csv(os.path.join(HERE, 'stop_logit_factor_20261015.csv')).set_index('modelo')
base = 'SC + VSC + used_two_compounds + TyreLife:C(Compound_Detail) + Position + GP + y21'
FORM = {'L2 tramo flexible': base + ' + bs(RaceFrac, df=5)',
        'L3 tramo propio de 2021': base + ' + bs(RaceFrac, df=5) + y21:bs(RaceFrac, df=5)',
        'L4 L3 + LapsLeft y SC/VSC 2021': base + ' + bs(RaceFrac, df=5) + y21:bs(RaceFrac, df=5) + y21:LapsLeft + y21:SC + y21:VSC'}


def clustered_se(name):
    f = lt.loc[name, 'f']
    d = train.assign(TyreLife=np.where(train.y21 == 1, tl0 * f, tl0))
    m = smf.logit('PitIn ~ ' + FORM[name], data=d).fit(disp=False, maxiter=300)
    slope = {c: m.params.get(f'TyreLife:C(Compound_Detail)[{c}]', 0.0) for c in d.Compound_Detail.unique()}
    d['z'] = d.y21 * tl0 * d.Compound_Detail.map(slope)
    m2 = smf.logit('PitIn ~ ' + FORM[name] + ' + z', data=d).fit(disp=False, maxiter=300, cov_type='cluster',
                                                                  cov_kwds={'groups': pd.factorize(d.race)[0]})
    return f + m2.params['z'], m2.bse['z']


PHYS = {'fisica, combustible por anio': (0.660, 0.072), 'fisica, curva comun': (0.555, 0.074)}
rows = []
for pn, bn in [('fisica, combustible por anio', 'L3 tramo propio de 2021'),
               ('fisica, combustible por anio', 'L4 L3 + LapsLeft y SC/VSC 2021'),
               ('fisica, curva comun', 'L2 tramo flexible')]:
    fp, sp = PHYS[pn]
    fb, sb = clustered_se(bn)
    w = np.array([1 / sp ** 2, 1 / sb ** 2])
    fj = (w * [fp, fb]).sum() / w.sum(); sj = w.sum() ** -0.5
    q = (fp - fb) ** 2 / (sp ** 2 + sb ** 2)
    rows.append(dict(fisica=pn, f_fis=fp, ee_fis=sp, conductual=bn, f_cond=fb, ee_cond=sb,
                     peso_fis=w[0] / w.sum(), f_conjunto=fj, ee=sj, ic95=f'{fj - 1.96 * sj:.2f}-{fj + 1.96 * sj:.2f}',
                     chi2_compat=q, p_compat=chi2.sf(q, 1)))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('\n=== Factor unico de TyreLife 2021 (fila 1 = especificacion principal)')
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(HERE, 'joint_factor_20261015.csv'), index=False)
