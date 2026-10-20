"""2026-10-14: evaluacion vuelta a vuelta de modelos de rivales (sin entrenar agentes).

Ajuste con 2021-23 (mismo preprocesamiento que src/models/rival_models.py, que reproduce exactamente los pickles del
simulador: reproduce_check_20261014.py); prueba fuera de muestra con 2024 (GP vistos en 2021-23).

Etapa 1, ¿para en esta vuelta? (logit binario)
  S0  simulador: SC + VSC + used_two + EarlyLaps + LateLaps + TyreLife:C(compuesto) + Position + GP
  S1  paper:     SC + VSC + used_two + LapsLeft + TyreLife:C(compuesto) + Position + GP
  S2  flexible:  SC + VSC + used_two + bs(fraccion de carrera, df=5) + TyreLife:C(compuesto) + Position + GP
  Cada uno sin y con las reglas fijas del simulador (+R): no para en vueltas 1-5, ni con <= 7 vueltas restantes en
  verde, stint minimo 8 en verde, y bajo SC/VSC solo en la primera vuelta del periodo. Con reglas, la probabilidad es 0
  en vueltas prohibidas (se acota a 1e-4 para el log-loss) y se informa el % de paradas reales que caen ahi.
Etapa 2, ¿que compuesto? (solo vueltas con parada y compuesto nuevo conocido, alternativas = compuestos del GP-anio)
  K0  simulador: logit binario sobre filas alternativa, chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change + GP,
      probabilidades normalizadas en el set
  K1  logit condicional (softmax) con C(alt) + C(alt):LapsLeft + not_change (sin GP: no se identifica)
  K1r logit condicional por rol del GP (duro/medio/blando): C(rol) + C(rol):LapsLeft + not_change
Metricas: log-loss medio (por vuelta / por decision), Brier, pseudo-R2 fuera de muestra frente a la tasa base de
2021-23, y calibracion (paradas esperadas vs observadas) por tramo de carrera y bandera.
Salida: esta carpeta (lap_eval_*.csv) y consola.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()  # noqa: E401
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
import patsy
from models import rival_models as R

PAPER_GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
             'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
FLOOR = 1e-4
pd.set_option('display.width', 220)


def prep(df):
    df = R.preprocess_laps(df)
    tot = df.groupby(['Year', 'GP'])['LapNumber'].transform('max')
    df['RaceFrac'] = df['LapNumber'] / tot
    df = df.sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
    # primera vuelta del periodo SC/VSC (por carrera; el estado de pista es comun a todos)
    st = df.groupby(['Year', 'GP', 'LapNumber'])[['SC', 'VSC']].max().reset_index()
    st = st.sort_values(['Year', 'GP', 'LapNumber'])
    for c in ['SC', 'VSC']:
        prev = st.groupby(['Year', 'GP'])[c].shift(1, fill_value=0)
        st[f'first_{c}'] = ((st[c] == 1) & (prev == 0)).astype(int)
    df = df.merge(st[['Year', 'GP', 'LapNumber', 'first_SC', 'first_VSC']], on=['Year', 'GP', 'LapNumber'], how='left')
    yf = (df.SC == 1) | (df.VSC == 1)
    first = (df.first_SC == 1) | (df.first_VSC == 1)
    df['allowed'] = ((df.LapNumber > 5) &
                     np.where(yf, first, (df.LapsLeft > 7) & (df.TyreLife >= 8))).astype(int)
    return df


train = prep(R.load_lap_data())
test = prep(pd.read_csv(os.path.join(R.DATA_PATH, 'session_2024_V2.csv')))
test = test[test.GP.isin(train.GP.unique())].copy()
print(f'entrenamiento 2021-23: {len(train)} vueltas, {int(train.PitIn.sum())} paradas | '
      f'prueba 2024: {len(test)} vueltas, {int(test.PitIn.sum())} paradas, {test.GP.nunique()} GP')

# ------------------------------------------------------------------ etapa 1
F = 'SC + VSC + used_two_compounds + {t} + TyreLife:C(Compound_Detail) + Position + GP'
STOP = {'S0': F.format(t='EarlyLaps + LateLaps'), 'S1': F.format(t='LapsLeft'),
        'S2': F.format(t='bs(RaceFrac, df=5)')}
base = train.PitIn.mean()


def ll(y, p):
    p = np.clip(p, FLOOR, 1 - FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


subsets = {'2024 todos': lambda d: d.index == d.index,
           '2024 8 GP del paper': lambda d: d.GP.isin(PAPER_GPS),
           '2024 sin vueltas 1-3': lambda d: d.LapNumber > 3}
rows, preds = [], {}
for name, f in STOP.items():
    fit = smf.logit('PitIn ~ ' + f, data=train).fit(disp=False, maxiter=200)
    p = np.asarray(fit.predict(test))
    for rules in (False, True):
        q = p * test.allowed.values if rules else p
        key = name + ('+R' if rules else '')
        preds[key] = q
        for sname, sf in subsets.items():
            m = np.asarray(sf(test))
            y = test.PitIn.values[m]; qq = q[m]
            L, L0 = ll(y, qq).mean(), ll(y, np.full_like(qq, base)).mean()
            rows.append(dict(modelo=key, subconjunto=sname, k=len(fit.params), aic_train=fit.aic,
                             logloss_x1000=1000 * L, brier_x1000=1000 * np.mean((y - qq) ** 2),
                             pseudoR2_test=1 - L / L0, esperadas=qq.sum(), observadas=int(y.sum()),
                             pct_paradas_prohibidas=100 * (y * (1 - test.allowed.values[m])).sum() / max(y.sum(), 1)
                             if rules else np.nan))
r1 = pd.DataFrame(rows)
r1.to_csv(os.path.join(HERE, 'lap_eval_stop_20261014.csv'), index=False)
print('\n=== Etapa 1 (parar o no), fuera de muestra 2024')
print(r1.round(3).to_string(index=False))

# calibracion por tramo de carrera y bandera (2024, todos)
test['tramo'] = pd.cut(test.RaceFrac, [0, .1, .2, .4, .6, .8, .9, 1.0])
test['bandera'] = np.where(test.SC == 1, 'SC', np.where(test.VSC == 1, 'VSC', 'verde'))
cal = test.groupby(['bandera', 'tramo'], observed=True).agg(vueltas=('PitIn', 'size'), obs=('PitIn', 'sum'))
for k, q in preds.items():
    cal[k] = pd.Series(q, index=test.index).groupby([test.bandera, test.tramo], observed=True).sum()
cal = cal.reset_index()
cal.to_csv(os.path.join(HERE, 'lap_eval_stop_calibration_20261014.csv'), index=False)
print('\n=== Calibracion etapa 1 (paradas observadas vs esperadas, 2024)')
print(cal.round(1).to_string(index=False))

# ------------------------------------------------------------------ etapa 2
ROLES = {}
for yr, gps in R.COMPOUND_MAPS.items():
    for gp, mp in gps.items():
        for role, c in mp.items():
            ROLES[(yr, gp, c)] = role


def long_df(d):
    d = d[(d.PitIn == 1) & d.NewCompound.notna()].copy()
    d['decision_id'] = d.Year.astype(str) + '_' + d.GP + '_' + d.Driver + '_' + d.LapNumber.astype(int).astype(str)
    alts = pd.DataFrame([dict(Year=y, GP=g, alt=c) for (y, g, c) in ROLES])
    L = d.merge(alts, on=['Year', 'GP'], how='inner')
    L['chosen'] = (L.alt == L.NewCompound).astype(int)
    L['not_change_compound'] = ((L.alt == L.Compound_Detail) & (L.used_two_compounds == 0)).astype(int)
    L['role'] = [ROLES[(y, g, c)] for y, g, c in zip(L.Year, L.GP, L.alt)]
    ok = L.groupby('decision_id').chosen.transform('sum') == 1          # compuesto nuevo dentro del set del GP
    return L[ok].sort_values(['decision_id', 'alt']).reset_index(drop=True)


Ltr, Lte = long_df(train), long_df(test)
print(f'\nEtapa 2: {Ltr.decision_id.nunique()} decisiones 2021-23, {Lte.decision_id.nunique()} decisiones 2024')


def softmax_by(v, g):
    v = pd.Series(v); e = np.exp(v - v.groupby(g).transform('max'))
    return (e / e.groupby(g).transform('sum')).values


res2 = {}
k0 = smf.logit('chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP', data=Ltr).fit(disp=False)
p = pd.Series(np.asarray(k0.predict(Lte)))
res2['K0'] = (p / p.groupby(Lte.decision_id.values).transform('sum')).values
for name, rhs, levels in [('K1', 'C(alt) + C(alt):LapsLeft + not_change_compound - 1', None),
                          ('K1r', 'C(role) + C(role):LapsLeft + not_change_compound - 1', None)]:
    Xtr = patsy.dmatrix(rhs, Ltr, return_type='dataframe')
    # en el logit condicional los efectos fijos de alternativa necesitan una referencia: se quita la primera columna
    Xtr = Xtr.drop(columns=[Xtr.columns[0]])
    fit = ConditionalLogit(Ltr.chosen.values, Xtr, groups=Ltr.decision_id.values).fit(disp=False, method='newton',
                                                                                         maxiter=200)
    Xte = patsy.build_design_matrices([patsy.dmatrix(rhs, Ltr).design_info], Lte, return_type='dataframe')[0]
    Xte = Xte[Xtr.columns]
    res2[name] = softmax_by(Xte.values @ fit.params.values, Lte.decision_id.values)
    print(f'{name}: coeficientes', dict(zip(Xtr.columns, np.round(fit.params.values, 3))))

rows = []
for name, pr in res2.items():
    for sname, m in [('2024 todos', np.ones(len(Lte), bool)), ('2024 8 GP del paper', Lte.GP.isin(PAPER_GPS).values)]:
        d = Lte[m].assign(p=pr[m])
        ch = d[d.chosen == 1]
        top = d.loc[d.groupby('decision_id').p.idxmax()]
        nalt = d.groupby('decision_id').size()
        rows.append(dict(modelo=name, subconjunto=sname, decisiones=len(ch),
                         logloss=-np.log(np.clip(ch.p, 1e-6, 1)).mean(),
                         logloss_uniforme=np.log(nalt).mean(),
                         acierto_top=(top.chosen == 1).mean()))
r2 = pd.DataFrame(rows)
r2.to_csv(os.path.join(HERE, 'lap_eval_compound_20261014.csv'), index=False)
print('\n=== Etapa 2 (compuesto), fuera de muestra 2024')
print(r2.round(3).to_string(index=False))
cal2 = Lte.assign(**{k: v for k, v in res2.items()}).groupby('role')[['chosen'] + list(res2)].sum()
print('\n=== Calibracion etapa 2 por rol (elecciones observadas vs esperadas, 2024)')
print(cal2.round(1).to_string())
cal2.to_csv(os.path.join(HERE, 'lap_eval_compound_calibration_20261014.csv'))
