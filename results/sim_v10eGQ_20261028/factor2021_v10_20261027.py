"""2026-10-27 — Factor unico de TyreLife 2021 con la definicion v10 (misma especificacion principal que
results/tyrelife_2021_20261015/joint_factor_20261015.py, fila 1):
  fisica:     Ec. 1 por GP con combustible por anio (laptime_factor_20261015.py, robustez), sin las vueltas de paso por el
              pit lane sin parada (common_v10_20261027.patch_regressions);
  conductual: logit de parada L3 (tramo de carrera propio de 2021), con paradas reales y POR EVENTO bajo SC/VSC, como el
              LogitStop v10 (rivals_v10_20261027.event_level); perfil de verosimilitud para f y ee agrupado por carrera;
  conjunta:   distancia minima (1/ee^2).
Control: CONTROL=1 (sin cambios v10) debe dar 0.660 / 0.786 / 0.716.
    F1_TYRELIFE_2021=none ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/factor2021_v10_20261027.py
Salida: results/sim_v10_20261027/factor2021_v10_20261027.csv
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
assert os.environ.get('F1_TYRELIFE_2021') == 'none', 'F1_TYRELIFE_2021=none'
os.environ['TLV'] = 'ninguno'
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__)); T21 = os.path.join(ROOT, 'results', 'tyrelife_2021_20261015')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'results', 'rival_model_eval_20261014'),
                os.path.join(ROOT, 'results', 'tyre_coef_study_20260930'), HERE]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
import models.regressions as R
from models import rival_models as RM
import common_v10_20261027 as CV
CONTROL = os.environ.get('CONTROL') == '1'

# ---------------- fisica
if not CONTROL:
    CV.patch_regressions(R)
src = open(os.path.join(T21, 'laptime_factor_20261015.py')).read()
pre = src.split('\nrows = []')[0]            # datos, clase GP, estimate(), GP_OBJ (base)
fuel = src.split('# ------------------------------------------------------------------ robustez: efecto combustible por anio')[1]
G = {'__file__': os.path.join(T21, 'laptime_factor_20261015.py'), '__name__': 'lf'}
exec(pre, G)
exec(fuel.split('print(')[0], G)                # FUEL, GP_OBJ con combustible por anio, fr, ser
f_fis, se_fis = float(G['fr'][('2021', '*')]), float(G['ser'][('2021', '*')])
print(f'fisica (combustible por anio): f = {f_fis:.3f}, ee {se_fis:.3f}', flush=True)

# ---------------- conductual
import rivals_common_20261014 as RC
if not CONTROL:
    _orig = RM.load_lap_data
    def _load():
        RM.load_lap_data = _orig
        try:
            return CV.rival_laps(RM).drop(columns=['PitIn_raw'])
        finally:
            RM.load_lap_data = _load
    RM.load_lap_data = _load
train, _ = RC.load()
if not CONTROL:
    import importlib.util
    spec = importlib.util.spec_from_file_location('r10', os.path.join(HERE, 'rivals_v10_20261027.py'))
    r10 = importlib.util.module_from_spec(spec)
    src10 = open(os.path.join(HERE, 'rivals_v10_20261027.py')).read()
    exec(src10.split('\ndef event_level')[0].split('assert os.environ')[0], r10.__dict__)
    exec('def event_level' + src10.split('\ndef event_level')[1].split('\ndef main')[0], r10.__dict__)
    train, _first = r10.event_level(train)
    train = train.reset_index(drop=True)
train['race'] = train.Year.astype(str) + '|' + train.GP
tl0 = train.TyreLife.values.copy()
FORM = ('SC + VSC + used_two_compounds + TyreLife:C(Compound_Detail) + Position + GP + y21'
        ' + bs(RaceFrac, df=5) + y21:bs(RaceFrac, df=5)')


def fitf(f, **kw):
    d = train.assign(TyreLife=np.where(train.y21 == 1, tl0 * f, tl0))
    return smf.logit('PitIn ~ ' + FORM, data=d).fit(disp=False, maxiter=300, **kw), d


r = minimize_scalar(lambda f: -fitf(f)[0].llf, bounds=(0.2, 1.6), method='bounded', options=dict(xatol=2e-3))
fb = r.x
m, d = fitf(fb)
slope = {c: m.params.get(f'TyreLife:C(Compound_Detail)[{c}]', 0.0) for c in d.Compound_Detail.unique()}
d['z'] = d.y21 * tl0 * d.Compound_Detail.map(slope)
m2 = smf.logit('PitIn ~ ' + FORM + ' + z', data=d).fit(disp=False, maxiter=300, cov_type='cluster',
                                                      cov_kwds={'groups': pd.factorize(d.race)[0]})
f_cond, se_cond = fb + m2.params['z'], m2.bse['z']
print(f'conductual (L3{", por evento, paradas reales" if not CONTROL else ""}): f = {f_cond:.3f}, ee {se_cond:.3f}', flush=True)

w = np.array([1 / se_fis ** 2, 1 / se_cond ** 2])
fj = (w * [f_fis, f_cond]).sum() / w.sum(); sj = w.sum() ** -0.5
q = (f_fis - f_cond) ** 2 / (se_fis ** 2 + se_cond ** 2)
row = dict(version='control' if CONTROL else 'v10', f_fis=f_fis, ee_fis=se_fis, f_cond=f_cond, ee_cond=se_cond,
           f_conjunto=fj, ee=sj, ic95=f'{fj - 1.96 * sj:.2f}-{fj + 1.96 * sj:.2f}', p_compat=chi2.sf(q, 1))
print(row)
if not CONTROL:
    pd.DataFrame([row]).to_csv(os.path.join(HERE, 'factor2021_v10_20261027.csv'), index=False)
