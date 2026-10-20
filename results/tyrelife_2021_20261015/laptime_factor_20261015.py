"""2026-10-15: estimacion "fisica" del factor de TyreLife 2021 con los tiempos de vuelta (Ec. 1) y validacion con 2024.

Datos: exactamente los de la Ec. 1 del simulador (R.all_laps + filtro A, como results/tyre_coef_study_20260930/
pooled_20261004.py), pero con TyreLife 2021 SIN reescalar (F1_TYRELIFE_2021=none). Muestra por GP: TrackStatus == 1 y
filtro IQR de R.filtrar_outliers_iqr, fijada con f = 1 (la muestra no cambia al variar f).

Modelo: Ec. 1 por GP (misma formula de R.calibrar_regresion_gp) con TyreLife_ef = TyreLife * f_c si la vuelta es de
2021 (f comun, o un f por compuesto). Los demas coeficientes son propios de cada GP. f se estima minimizando
sum_g SSR_g / s2_g (s2_g de la Ec. 1 con f = 1). Error estandar: linealizacion (Gauss-Newton) con errores agrupados por
carrera (anio x GP), porque los residuos de un mismo piloto/carrera estan correlacionados.
Variantes: f comun; f por compuesto; f21 y f23 (2023 relativo a 2022) para ver si 2022 y 2023 son homogeneos.

Validacion 2024 (fuera de muestra): forma de la degradacion dentro de cada stint. Prediccion de la Ec. 1 (2021-23 con
cada f) para las vueltas limpias 2024 sin pit, menos su media en el stint, frente al tiempo real menos su media en el
stint (asi se cancelan piloto, anio, compuesto y nivel). Metrica: RMSE y error absoluto medio.
    F1_TYRELIFE_2021=none python results/tyrelife_2021_20261015/laptime_factor_20261015.py
"""
import io, os, sys, contextlib, json
assert os.environ.get('F1_TYRELIFE_2021') == 'none', 'correr con F1_TYRELIFE_2021=none'
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'results', 'tyre_coef_study_20260930'))
os.chdir(ROOT)
import config, compat; compat.setup_all()  # noqa: E401
import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
import models.regressions as R
import pooled_20261004 as P

pd.set_option('display.width', 200)
H = P.hist_laps_A()
GPS_ALL = sorted(H.GP.unique())
BASE = ("LapTime ~ LapNumber + LapNumber_2 + FirstLap_pos + Interval_front + Interval_behind + PitIn + PitOut + DRS "
        "+ C(Driver, Treatment(reference='VER')) + C(Compound_Detail, Treatment(reference='{ref}')) "
        "+ C(Year, Treatment(reference='{yref}'))")
COMPS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']


class GP:
    """Ec. 1 de un GP con la muestra fija; TyreLife x compuesto se arma a mano para poder escalar 2021 y 2023."""
    def __init__(self, gp, base=None):
        base = base or BASE
        with contextlib.redirect_stdout(io.StringIO()):
            m, d = R.filtrar_outliers_iqr(H, gp)
        d = d.copy(); d['Year'] = d.Year.astype(int).astype(str)
        self.gp, self.d = gp, d
        ref = d.Compound_Detail.value_counts().idxmax(); yref = sorted(d.Year.unique())[0]
        self.X0 = np.asarray(patsy.dmatrix(base.split('~')[1].format(ref=ref, yref=yref), d))
        self.y = d.LapTime.values
        self.comps = sorted(d.Compound_Detail.unique())
        self.D = np.column_stack([(d.Compound_Detail == c).values for c in self.comps]).astype(float)
        self.tl = d.TyreLife.values; self.yr = d.Year.values
        self.race = (d.Year + '|' + gp).values
        self.n21 = int((self.yr == '2021').sum())
        self.s2 = self.fit({})[1] / (len(self.y) - self.X0.shape[1] - self.D.shape[1])

    def mult(self, f):
        """f: dict {('2021', comp) o ('2021', '*') o ('2023', '*'): factor}."""
        m = np.ones(len(self.y))
        for (yr, c), v in f.items():
            sel = self.yr == yr
            if c != '*':
                sel &= (self.d.Compound_Detail.values == c)
            m[sel] = v
        return m

    def fit(self, f):
        X = np.column_stack([self.X0, self.D * (self.tl * self.mult(f))[:, None]])
        b, *_ = np.linalg.lstsq(X, self.y, rcond=None)
        e = self.y - X @ b
        return b, e @ e, X, e


GP_OBJ = {g: GP(g) for g in GPS_ALL}
GPS21 = [g for g in GPS_ALL if GP_OBJ[g].n21 > 0 and (GP_OBJ[g].yr != '2021').any()]
print(f'{len(GPS_ALL)} GP con Ec. 1; {len(GPS21)} con vueltas de 2021 y de 2022-23; '
      f'vueltas 2021: {sum(GP_OBJ[g].n21 for g in GPS21)}')


def estimate(keys, gps, x0=None):
    """keys: lista de claves de f a estimar. Devuelve f, ee (agrupado por carrera)."""
    def obj(x):
        f = dict(zip(keys, x))
        return sum(GP_OBJ[g].fit(f)[1] / GP_OBJ[g].s2 for g in gps)
    r = minimize(obj, x0 if x0 is not None else np.ones(len(keys)), method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
    f = dict(zip(keys, r.x))
    # Gauss-Newton: derivada de las predicciones respecto de cada f_k, parcializada por el resto de la Ec. 1
    A = np.zeros((len(keys), len(keys))); meat = {}
    for g in gps:
        o = GP_OBJ[g]; b, _, X, e = o.fit(f)
        bs = b[o.X0.shape[1]:]                                   # pendientes por compuesto
        J = []
        for (yr, c) in keys:
            sel = (o.yr == yr) & ((o.d.Compound_Detail.values == c) if c != '*' else True)
            J.append(np.where(sel, (o.D @ bs) * o.tl, 0.0))
        J = np.column_stack(J)
        Q, _ = np.linalg.qr(X)
        Jt = J - Q @ (Q.T @ J)                                   # parcializado
        w = 1 / o.s2
        A += w * Jt.T @ Jt
        for rc in np.unique(o.race):
            s = o.race == rc
            meat[rc] = w * Jt[s].T @ e[s]
    Ainv = np.linalg.pinv(A)
    Bm = sum(np.outer(v, v) for v in meat.values())
    se = np.sqrt(np.diag(Ainv @ Bm @ Ainv)); se_ols = np.sqrt(np.diag(Ainv))
    return f, dict(zip(keys, se)), dict(zip(keys, se_ols)), r.fun


rows = []
f, se, se0, _ = estimate([('2021', '*')], GPS21)
rows.append(('comun', '2021 todos', f[('2021', '*')], se[('2021', '*')], se0[('2021', '*')]))
comp_keys = [('2021', c) for c in COMPS if any((GP_OBJ[g].d[GP_OBJ[g].yr == '2021'].Compound_Detail == c).any()
                                                for g in GPS21)]
fc, sec, sec0, _ = estimate(comp_keys, GPS21)
for k in comp_keys:
    n = sum(int(((GP_OBJ[g].yr == '2021') & (GP_OBJ[g].d.Compound_Detail.values == k[1])).sum()) for g in GPS21)
    rows.append(('por compuesto', f'2021 {k[1]} ({n} vueltas)', fc[k], sec[k], sec0[k]))
GPS23 = [g for g in GPS21 if (GP_OBJ[g].yr == '2022').any() and (GP_OBJ[g].yr == '2023').any()]
fh, seh, seh0, _ = estimate([('2021', '*'), ('2023', '*')], GPS23)
for k in fh:
    rows.append(('homogeneidad (ref. 2022)', f'{k[0]} ({len(GPS23)} GP)', fh[k], seh[k], seh0[k]))
T = pd.DataFrame(rows, columns=['modelo', 'parametro', 'f', 'ee_agrupado', 'ee_ols'])
T['ic95'] = [f'{a - 1.96 * s:.2f}-{a + 1.96 * s:.2f}' for a, s in zip(T.f, T.ee_agrupado)]
print('\n=== Factor de TyreLife (pendiente relativa a 2022-23) estimado con tiempos de vuelta, Ec. 1 por GP')
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(HERE, 'laptime_factor_20261015.csv'), index=False)

# ------------------------------------------------------------------ validacion 2024
L24 = P.laps_2024_A()
L24 = L24[(L24.TrackStatus.astype(str).isin(['1', '1.0'])) & (L24.PitIn == 0) & (L24.PitOut == 0) & (L24.LapNumber > 1)]
LEG = {'C0': 0.65, 'C1': 1.0, 'C2': 0.90, 'C3': 0.98, 'C4': 0.92, 'C5': 0.92}
VARS = {'legacy': {('2021', c): v for c, v in LEG.items()}, 'ninguno': {},
        'conductual 0.84': {('2021', '*'): 0.84}, 'fisico robusto 0.66': {('2021', '*'): 0.66},
        'fisico comun': f, 'fisico por compuesto': fc}
SHAPE = ['LapNumber', 'LapNumber_2', 'FirstLap_pos', 'Interval_front', 'Interval_behind', 'DRS']
out = []
for g in GPS_ALL:
    o = GP_OBJ[g]; t = L24[L24.GP == g].copy()
    t = t[t.Compound_Detail.isin(o.comps)]
    if len(t) < 50:
        continue
    t['key'] = t.Driver + '|' + t.stint.astype(str)
    names = patsy.dmatrix(BASE.split('~')[1].format(ref=o.d.Compound_Detail.value_counts().idxmax(),
                                                    yref=sorted(o.d.Year.unique())[0]), o.d).design_info.column_names
    for v, fv in VARS.items():
        b = o.fit(fv)[0]
        bb = dict(zip(names, b[:len(names)]))
        sl = dict(zip(o.comps, b[len(names):]))
        pred = sum(bb[k] * t[k].values for k in SHAPE) + t.TyreLife.values * t.Compound_Detail.map(sl).values
        t['p'] = pred
        dm = (t.LapTime - t.groupby('key').LapTime.transform('mean')) - (t.p - t.groupby('key').p.transform('mean'))
        dm = dm[t.groupby('key').LapTime.transform('size') >= 5]
        dm = dm[dm.abs() < 5]                                    # trafico/errores grandes fuera
        out.append(dict(GP=g, variante=v, vueltas=len(dm), rmse=np.sqrt((dm ** 2).mean()), mae=dm.abs().mean(),
                        paper=g in P.PAPER))
V = pd.DataFrame(out)
V.to_csv(os.path.join(HERE, 'laptime_validation_2024_20261015.csv'), index=False)
print('\n=== Validacion 2024: forma de la degradacion dentro del stint (s/vuelta; menor es mejor)')
for lab, m in [('todos los GP', V.GP == V.GP), ('8 GP del paper', V.paper)]:
    s = V[m].groupby('variante', sort=False).apply(lambda d: pd.Series(dict(
        GP=d.GP.nunique(), rmse=np.sqrt((d.rmse ** 2 * d.vueltas).sum() / d.vueltas.sum()),
        mae=(d.mae * d.vueltas).sum() / d.vueltas.sum())))
    print(lab); print(s.round(4).to_string())

# ------------------------------------------------------------------ validacion 2024 (2): pendientes por GP x compuesto
# Ec. 1 con la carrera 2024 sola (vara, como pooled_20261004) frente a la pendiente 2021-23 con cada f.
Y24 = {}; L24A = P.laps_2024_A()
for g in GPS_ALL:
    try:
        Y24[g] = P.fit_gp(L24A, g) if g in set(L24.GP) else None
    except Exception as ex:  # noqa: BLE001
        print('2024 sin regresion:', g, ex)
rows = []
for g, r in Y24.items():
    if r is None:
        continue
    o = GP_OBJ[g]
    sl24 = {k[1]: (b, np.sqrt(r['V'][i, i])) for i, (k, b) in enumerate(zip(r['kinds'], r['b'])) if k[0] == 'sl'}
    for v, fv in VARS.items():
        b = o.fit(fv)[0][o.X0.shape[1]:]
        for c, bh in zip(o.comps, b):
            if c in sl24 and r['stints'].get(c, 0) >= 5:
                rows.append(dict(GP=g, comp=c, variante=v, h2123=bh, v24=sl24[c][0], se24=sl24[c][1],
                                 paper=g in P.PAPER))
S = pd.DataFrame(rows); S['err'] = S.h2123 - S.v24
S.to_csv(os.path.join(HERE, 'laptime_slope_validation_2024_20261015.csv'), index=False)
print('\n=== Validacion 2024 (2): pendiente de TyreLife 2021-23 (con cada f) frente a la Ec. 1 de 2024 sola (s/vuelta)')
for lab, m in [('todos los GP', S.GP == S.GP), ('8 GP del paper', S.paper)]:
    print(lab, f'({S[m].groupby("variante").size().iloc[0]} celdas GP x compuesto)')
    print(S[m].groupby('variante', sort=False).apply(lambda d: pd.Series(dict(
        sesgo=d.err.mean(), MAE=d.err.abs().mean(),
        MAE_pond=np.average(d.err.abs(), weights=1 / d.se24 ** 2)))).round(4).to_string())

# ------------------------------------------------------------------ robustez: efecto combustible por anio
FUEL = BASE.replace('LapNumber + LapNumber_2', 'LapNumber + LapNumber_2 + LapNumber:C(Year) + LapNumber_2:C(Year)')
GP_OBJ = {g: GP(g, FUEL) for g in GPS21}
fr, ser, _, _ = estimate([('2021', '*')], GPS21)
print(f"\n=== Robustez (LapNumber y LapNumber^2 por anio): f comun 2021 = {fr[('2021', '*')]:.3f} "
      f"(ee agrupado {ser[('2021', '*')]:.3f})")
