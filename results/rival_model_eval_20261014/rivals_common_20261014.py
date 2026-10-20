"""2026-10-14: piezas comunes de la evaluacion de modelos de rivales (datos, modelos S0-S3 y K0-K1, reglas).

Datos: mismo preprocesamiento que src/models/rival_models.py (reproduce los pickles del simulador). 2021-23 = ajuste,
2024 = prueba. Se agregan: RaceFrac (vuelta / vueltas de la carrera), YF (SC o VSC), periodo YF (vueltas consecutivas
con SC o VSC en la carrera) y su primera vuelta.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()  # noqa: E401,F401
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
import patsy
from models import rival_models as R

PAPER_GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
             'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']

_F = 'SC + VSC + used_two_compounds + {t} + TyreLife:C(Compound_Detail) + Position + GP'
STOP = {
    'S0': _F.format(t='EarlyLaps + LateLaps'),                       # simulador
    'S1': _F.format(t='LapsLeft'),                                   # paper
    'S2': _F.format(t='bs(RaceFrac, df=5)'),                         # flexible
    # S3: S2 + la forma en la carrera depende de si ya uso dos compuestos, y bajo SC/VSC del tramo de carrera
    'S3': _F.format(t='bs(RaceFrac, df=5) + used_two_compounds:bs(RaceFrac, df=4) + YF:bs(RaceFrac, df=3)'),
}

ROLES = {(yr, gp, c): role for yr, gps in R.COMPOUND_MAPS.items() for gp, mp in gps.items() for role, c in mp.items()}


def prep(df):
    df = R.preprocess_laps(df)
    tot = df.groupby(['Year', 'GP'])['LapNumber'].transform('max')
    df['NLaps'] = tot
    df['RaceFrac'] = df['LapNumber'] / tot
    df['YF'] = ((df.SC == 1) | (df.VSC == 1)).astype(int)
    st = (df.groupby(['Year', 'GP', 'LapNumber'])['YF'].max().reset_index()
            .sort_values(['Year', 'GP', 'LapNumber']))
    prev = st.groupby(['Year', 'GP'])['YF'].shift(1, fill_value=0)
    prev_lap = st.groupby(['Year', 'GP'])['LapNumber'].shift(1)
    st['yf_first'] = ((st.YF == 1) & ((prev == 0) | (st.LapNumber - prev_lap > 1))).astype(int)
    st['yf_period'] = np.where(st.YF == 1, st.groupby(['Year', 'GP'])['yf_first'].cumsum(), 0)
    df = df.merge(st[['Year', 'GP', 'LapNumber', 'yf_first', 'yf_period']], on=['Year', 'GP', 'LapNumber'], how='left')
    return df.sort_values(['Year', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)


# 2026-10-15: variante del reescalamiento de TyreLife 2021 (env TLV; por defecto el vigente del simulador).
#   vigente: R.TYRELIFE_ADJUST | ninguno: sin reescalar | comun: x0.84 (todos los compuestos) | M2: por compuesto
#   inter: sin reescalar y pendiente propia de 2021 (TyreLife:C(compuesto):y21); en 2024 se usa y21 = 0
#   Factores: results/tyrelife_2021_20261015/stint_factors_20261015.py
TLV = os.environ.get('TLV', 'vigente')
_TL = {'ninguno': dict.fromkeys(['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], 1.0),
       'comun': dict.fromkeys(['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], 0.84),
       'M2': {'C0': 0.84, 'C1': 1.0, 'C2': 0.86, 'C3': 0.86, 'C4': 0.70, 'C5': 0.785}}
_TL['inter'] = _TL['ninguno']
_TL['sin2021'] = _TL['ninguno']                               # sin2021: ajuste solo con 2022-23
_C21 = ['C0', 'C2', 'C3', 'C4', 'C5']                          # compuestos presentes en 2021
TLSUF = '' if TLV == 'vigente' else '_tl' + TLV                  # sufijo de archivos de salida


def load():
    if TLV != 'vigente':
        R.TYRELIFE_ADJUST = _TL[TLV]
    train = prep(R.load_lap_data())
    if TLV == 'sin2021':
        train = train[train.Year >= 2022].reset_index(drop=True)
    test = prep(pd.read_csv(os.path.join(R.DATA_PATH, 'session_2024_V2.csv')))
    test = test[test.GP.isin(train.GP.unique())].reset_index(drop=True)
    train['y21'] = (train.Year == 2021).astype(int); test['y21'] = 0
    for c in _C21:                                                # pendiente propia de 2021 (variante inter)
        train[f'tl21_{c}'] = train.TyreLife * train.y21 * (train.Compound_Detail == c); test[f'tl21_{c}'] = 0.0
    return train, test


def fit_stop(train):
    tl = 'TyreLife:C(Compound_Detail)'
    extra = ' + ' + ' + '.join(f'tl21_{c}' for c in _C21)
    st = {k: f.replace(tl, tl + extra) if TLV == 'inter' else f for k, f in STOP.items()}
    return {k: smf.logit('PitIn ~ ' + f, data=train).fit(disp=False, maxiter=300) for k, f in st.items()}


# --- reglas ---------------------------------------------------------------------------------------------------
# R  (simulador): verde: vuelta > 5, > 7 vueltas restantes, stint >= 8; SC/VSC: solo la primera vuelta del periodo
#    (y vuelta > 5).
# R2 (minima): verde: vuelta > 2; SC/VSC: sin restriccion.
# R3 (simulador en verde, libre bajo SC/VSC): verde como R; SC/VSC: cualquier vuelta del periodo.
def green_allowed(lap, laps_left, tyrelife, rules):
    if rules in ('R', 'R3'):
        return (lap > 5) & (laps_left > 7) & (tyrelife >= 8)
    if rules == 'R2':
        return lap > 2
    return np.ones_like(lap, dtype=bool)


# --- etapa 2 ------------------------------------------------------------------------------------------------
def long_df(d):
    d = d[(d.PitIn == 1) & d.NewCompound.notna()].copy()
    d['decision_id'] = d.Year.astype(str) + '_' + d.GP + '_' + d.Driver + '_' + d.LapNumber.astype(int).astype(str)
    alts = pd.DataFrame([dict(Year=y, GP=g, alt=c) for (y, g, c) in ROLES])
    L = d.merge(alts, on=['Year', 'GP'], how='inner')
    L['chosen'] = (L.alt == L.NewCompound).astype(int)
    L['not_change_compound'] = ((L.alt == L.Compound_Detail) & (L.used_two_compounds == 0)).astype(int)
    ok = L.groupby('decision_id').chosen.transform('sum') == 1
    return L[ok].sort_values(['decision_id', 'alt']).reset_index(drop=True)


# K1: el efecto de LapsLeft comun a todas las alternativas no se identifica; referencia = C3 (sin interaccion)
K1_RHS = ('C(alt, Treatment("C3")) + C(alt, Treatment("C3")):LapsLeft + not_change_compound - 1')


class K0:
    def __init__(self, Ltr):
        self.fit = smf.logit('chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP', data=Ltr).fit(disp=False)

    def probs(self, L):
        p = pd.Series(np.asarray(self.fit.predict(L)), index=L.index)
        return (p / p.groupby(L.decision_id).transform('sum')).values


class K1:
    def __init__(self, Ltr):
        X = patsy.dmatrix(K1_RHS, Ltr, return_type='dataframe')
        keep = [c for c in X.columns if not (c.endswith('[C3]') or c.endswith('[C3]:LapsLeft'))
                and X[c].abs().sum() > 0]
        # sin intercepto de alternativa para C3 (referencia) ni interaccion C3:LapsLeft
        X = X[[c for c in keep if 'T.C3' not in c]]
        self.cols, self.info = list(X.columns), patsy.dmatrix(K1_RHS, Ltr).design_info
        self.fit = ConditionalLogit(Ltr.chosen.values, X, groups=Ltr.decision_id.values).fit(
            disp=False, method='newton', maxiter=200)

    def probs(self, L):
        X = patsy.build_design_matrices([self.info], L, return_type='dataframe')[0][self.cols]
        v = pd.Series(X.values @ self.fit.params.values, index=L.index)
        e = np.exp(v - v.groupby(L.decision_id).transform('max'))
        return (e / e.groupby(L.decision_id).transform('sum')).values
