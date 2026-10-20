"""2026-10-14 (v2): evaluacion fuera de muestra (ajuste 2021-23, prueba 2024) de la etapa 1 (parar o no) y la etapa 2
(compuesto), con los periodos de SC/VSC como unidad (criterio del autor: si el auto para en cualquier vuelta del periodo y
el modelo dice que para en la primera, es acierto).

Unidades de la etapa 1:
  - vuelta en verde: y = PitIn de la vuelta, p = probabilidad del modelo en la vuelta (0 si la regla la prohibe);
  - piloto x periodo SC/VSC: y = paro en alguna vuelta del periodo;
      sin reglas y con R2: p = 1 - prod(1 - p_vuelta) sobre el periodo;
      con R (simulador):   p = probabilidad en la primera vuelta del periodo (0 si vuelta <= 5).
Reglas: ver rivals_common_20261014.py (R = simulador, R2 = minima, '' = ninguna).
Salida: lap_eval_v2_*.csv y lap_eval_v2_20261014.log en esta carpeta.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rivals_common_20261014 import *  # noqa: F401,F403
import rivals_common_20261014 as C

FLOOR = 1e-4
pd.set_option('display.width', 230)
train, test = C.load()
fits = C.fit_stop(train)
print(f'ajuste 2021-23: {len(train)} vueltas | prueba 2024: {len(test)} vueltas, {int(test.PitIn.sum())} paradas, '
      f'{test.GP.nunique()} GP; periodos SC/VSC 2024: {test[test.YF == 1].groupby(["GP", "yf_period"]).ngroups}')
print({k: round(f.aic, 1) for k, f in fits.items()}, '(AIC en 2021-23)')


def units(d, p, rules):
    """Devuelve DataFrame de unidades (y, p, tipo, tramo, GP, prohibida)."""
    d = d.assign(p=p)
    g = d[d.YF == 0]
    ok = C.green_allowed(g.LapNumber.values, g.LapsLeft.values, g.TyreLife.values, rules)
    ug = pd.DataFrame(dict(y=g.PitIn.values, p=g.p.values * ok, tipo='verde', frac=g.RaceFrac.values,
                           GP=g.GP.values, prohibida=~ok))
    y = d[d.YF == 1]
    grp = y.groupby(['Year', 'GP', 'Driver', 'yf_period'])
    first = grp.head(1).set_index(['Year', 'GP', 'Driver', 'yf_period'])
    uy = pd.DataFrame(dict(y=grp.PitIn.max(), p_any=1 - grp.p.apply(lambda s: np.prod(1 - s.values)),
                           frac=grp.RaceFrac.min()))
    uy['p_first'] = first.p * (first.LapNumber > 5)
    uy['p'] = uy.p_first if rules == 'R' else uy.p_any
    uy['prohibida'] = (first.LapNumber <= 5) if rules == 'R' else False
    uy = uy.reset_index()[['y', 'p', 'frac', 'GP', 'prohibida']].assign(tipo='SC/VSC')
    return pd.concat([ug, uy], ignore_index=True)


def ll(y, p):
    p = np.clip(p, FLOOR, 1 - FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


rows, cal = [], []
base_g = train[train.YF == 0].PitIn.mean()
for name, fit in fits.items():
    p = np.asarray(fit.predict(test))
    for rules in ['', 'R', 'R2', 'R3']:
        U = units(test, p, rules)
        key = name + ('+' + rules if rules else '')
        for sub, m in [('22 GP', np.ones(len(U), bool)), ('8 GP paper', U.GP.isin(C.PAPER_GPS).values)]:
            for tipo in ['total', 'verde', 'SC/VSC']:
                mm = m & ((U.tipo == tipo).values if tipo != 'total' else True)
                u = U[mm]
                rows.append(dict(modelo=key, sub=sub, tipo=tipo, unidades=len(u),
                                 logloss_x1000=1000 * ll(u.y, u.p).mean(),
                                 esperadas=u.p.sum(), observadas=int(u.y.sum()),
                                 pct_obs_prohibidas=100 * (u.y * u.prohibida).sum() / max(u.y.sum(), 1)))
        U['tramo'] = pd.cut(U.frac, [0, .1, .2, .4, .6, .8, .9, 1.0])
        c = U.groupby(['tipo', 'tramo'], observed=True).agg(unid=('y', 'size'), obs=('y', 'sum'), esp=('p', 'sum'))
        cal.append(c.esp.rename(key))
        if name == 'S0' and rules == '':
            cal.insert(0, c[['unid', 'obs']])
r1 = pd.DataFrame(rows)
r1.to_csv(os.path.join(C.HERE, f'lap_eval_v2_stop_20261014{C.TLSUF}.csv'), index=False)
print('\n=== Etapa 1, log-loss x1000 por unidad (menor es mejor), 2024')
print(r1.pivot_table(index='modelo', columns=['sub', 'tipo'], values='logloss_x1000', sort=False).round(1).to_string())
print('\n=== Etapa 1, paradas esperadas / observadas, 22 GP')
t = r1[r1['sub'] == '22 GP'].pivot_table(index='modelo', columns='tipo', values='esperadas', sort=False).round(0)
t['obs_total'] = r1[(r1['sub'] == '22 GP') & (r1.tipo == 'total')].set_index('modelo').observadas
t['%obs_prohibidas'] = r1[(r1['sub'] == '22 GP') & (r1.tipo == 'total')].set_index('modelo').pct_obs_prohibidas.round(1)
print(t.to_string())
calt = pd.concat(cal, axis=1).reset_index()
calt.to_csv(os.path.join(C.HERE, f'lap_eval_v2_stop_calibration_20261014{C.TLSUF}.csv'), index=False)
print('\n=== Calibracion etapa 1 (observadas vs esperadas por tipo de unidad y tramo de carrera), 2024')
cols = ['tipo', 'tramo', 'unid', 'obs'] + [c for c in calt.columns if c.startswith('S') and ('+R' not in c or c.endswith('R2'))]
print(calt[cols].round(1).to_string(index=False))

# ------------------------------------------------------------------ etapa 2
Ltr, Lte = C.long_df(train), C.long_df(test)
models = {'K0': C.K0(Ltr), 'K1': C.K1(Ltr)}
print('\nK1 (logit condicional, referencia C3):', dict(zip(models['K1'].cols, np.round(models['K1'].fit.params.values, 3))))
rows = []
for name, m in models.items():
    pr = m.probs(Lte)
    for sub, mask in [('22 GP', np.ones(len(Lte), bool)), ('8 GP paper', Lte.GP.isin(C.PAPER_GPS).values)]:
        d = Lte[mask].assign(p=pr[mask])
        ch = d[d.chosen == 1]
        top = d.loc[d.groupby('decision_id').p.idxmax()]
        rows.append(dict(modelo=name, sub=sub, decisiones=len(ch), logloss=-np.log(np.clip(ch.p, 1e-6, 1)).mean(),
                         logloss_uniforme=np.log(d.groupby('decision_id').size()).mean(), acierto=(top.chosen == 1).mean()))
    Lte[name] = pr
r2 = pd.DataFrame(rows)
r2.to_csv(os.path.join(C.HERE, f'lap_eval_v2_compound_20261014{C.TLSUF}.csv'), index=False)
print('\n=== Etapa 2 (compuesto), 2024')
print(r2.round(3).to_string(index=False))
Lte['role'] = [C.ROLES[(y, g, c)] for y, g, c in zip(Lte.Year, Lte.GP, Lte.alt)]
print(Lte.groupby('role')[['chosen', 'K0', 'K1']].sum().round(1).to_string())
