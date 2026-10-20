"""2026-10-15: factor de TyreLife 2021 estimado dentro del logit de paradas (analogo de laptime_factor: perfil de f).

TyreLife_ef = TyreLife * f en 2021; se busca el f que maximiza la verosimilitud del logit de parada de los rivales
(2021-23, datos de rival_models con TyreLife sin reescalar). Siempre con efecto propio de 2021 (y21, como las dummies
de anio de la Ec. 1). Variantes:
  L1  S0 del simulador (EarlyLaps + LateLaps) + y21
  L2  tramo de carrera flexible (bs(RaceFrac, 5)) + y21
  L3  L2 con la curva de tramo de carrera propia de 2021 (y21:bs(RaceFrac, 5)): analogo de la curva de combustible
      por anio del 0.66
  L4  L3 + LapsLeft propio de 2021 y SC/VSC propios de 2021
IC95: perfil de verosimilitud (caida de 1.92), sin agrupar por carrera (optimista).
"""
import os, sys
os.environ['TLV'] = 'ninguno'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rival_model_eval_20261014'))
import rivals_common_20261014 as RC
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

train, _ = RC.load()
tl0 = train.TyreLife.values.copy()
base = 'SC + VSC + used_two_compounds + TyreLife:C(Compound_Detail) + Position + GP + y21'
M = {'L1 S0 + y21': base + ' + EarlyLaps + LateLaps',
     'L2 tramo flexible': base + ' + bs(RaceFrac, df=5)',
     'L3 tramo propio de 2021': base + ' + bs(RaceFrac, df=5) + y21:bs(RaceFrac, df=5)',
     'L4 L3 + LapsLeft y SC/VSC 2021': base + ' + bs(RaceFrac, df=5) + y21:bs(RaceFrac, df=5) + y21:LapsLeft + y21:SC + y21:VSC'}


def ll(f, form):
    d = train.assign(TyreLife=np.where(train.y21 == 1, tl0 * f, tl0))
    return smf.logit('PitIn ~ ' + form, data=d).fit(disp=False, maxiter=300).llf


rows = []
for name, form in M.items():
    r = minimize_scalar(lambda f: -ll(f, form), bounds=(0.2, 1.6), method='bounded', options=dict(xatol=2e-3))
    fh, lmax = r.x, -r.fun
    lo = minimize_scalar(lambda f: abs(lmax - ll(f, form) - 1.92), bounds=(0.1, fh), method='bounded').x
    hi = minimize_scalar(lambda f: abs(lmax - ll(f, form) - 1.92), bounds=(fh, 3.0), method='bounded').x
    rows.append(dict(modelo=name, f=fh, ic95=f'{lo:.2f}-{hi:.2f}', llf=lmax, ll_f1=ll(1.0, form)))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_logit_factor_20261015.csv'), index=False)
