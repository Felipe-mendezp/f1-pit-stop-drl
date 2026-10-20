"""2026-10-04 — Calibracion conjunta (pooled) de coeficientes de neumatico y validacion contra 2024.

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/pooled_20261004.py
Salida (nada fuera de esta carpeta): results/tyre_coef_study_20260930/pooled_20261004/

1. Ec. 1 por GP con 2021-23 (all_laps() + filtro (A): se excluyen ediciones con >10 % de vueltas
   WET/INT) y con la carrera 2024 sola (vara de validacion). Se guardan los coeficientes de
   compuesto con su covarianza y el numero de stints por compuesto.
2. Modelo conjunto sobre todos los GP del calendario 2024 con historia, ajustado por minimos
   cuadrados generalizados (covarianza OLS de cada GP) + penalizaciones:
     fijos     o_gc = beta_c + v_g*(c-3) + e_gc                 (solo diferencias dentro del GP)
     variables s_gc = exp(theta_g + kappa_c + u_g*(c-3)) + e_gc
   Variante monotona: beta_c (kappa_c) = suma acumulada de pasos positivos, escalada por un
   factor positivo por GP (lambda_g = exp(.)), sin efecto de celda; el paso C4->C5 se fija igual
   al paso C3->C4 (los datos 2021-23 no informan ese paso: sin el supuesto colapsa a 0).
3. Opciones exportadas para el simulador:
     opcion2 = modelo libre con efecto de celda (tau_e = 0.1 s fijo, 0.01 s/vuelta variable);
               pendientes acotadas en >= 0.
     opcion3 = modelo monotono estructural (tau_e = 0), paso C4->C5 = paso C3->C4.
   La opcion 1 de referencia es la calibracion actual (data/completed_coefs_V2_filled.xlsx).
"""
import io, os, sys, contextlib
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import models.regressions as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pooled_20261004')
os.makedirs(OUT, exist_ok=True)

MAP = {'Bahrain': 'C1 C2 C3', 'Saudi Arabian': 'C2 C3 C4', 'Australian': 'C3 C4 C5', 'Japanese': 'C1 C2 C3',
       'Miami': 'C2 C3 C4', 'Monaco': 'C3 C4 C5', 'Canadian': 'C3 C4 C5', 'Spanish': 'C1 C2 C3',
       'Austrian': 'C3 C4 C5', 'British': 'C1 C2 C3', 'Hungarian': 'C3 C4 C5', 'Belgian': 'C2 C3 C4',
       'Dutch': 'C1 C2 C3', 'Italian': 'C3 C4 C5', 'Azerbaijan': 'C3 C4 C5', 'Singapore': 'C3 C4 C5',
       'United States': 'C2 C3 C4', 'Mexico City': 'C3 C4 C5', 'São Paulo': 'C3 C4 C5',
       'Las Vegas': 'C3 C4 C5', 'Abu Dhabi': 'C3 C4 C5', 'Emilia Romagna': 'C3 C4 C5', 'Chinese': 'C2 C3 C4'}
MAP = {k + ' Grand Prix': v.split() for k, v in MAP.items()}
PAPER = [g + ' Grand Prix' for g in ['Bahrain', 'Belgian', 'Dutch', 'Hungarian', 'Miami',
                                     'Saudi Arabian', 'Singapore', 'United States']]


# --------------------------------------------------------------------------- datos
def wet_editions(raw):
    share = raw.groupby(['Year', 'GP'])['Compound'].apply(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
    return set(share[share > 0.10].index)


def add_stint(d, raw):
    """stint = n.o de PitOut previos del piloto en esa carrera (sobre los datos crudos)."""
    raw = raw.sort_values(['Year', 'GP', 'Driver', 'LapNumber']).copy()
    raw['stint'] = raw.groupby(['Year', 'GP', 'Driver'])['PitOut'].cumsum()
    key = ['Year', 'GP', 'Driver', 'LapNumber']
    return d.merge(raw[key + ['stint']].drop_duplicates(key), on=key, how='left')


def hist_laps_A():
    """all_laps() (exactamente como el pipeline) + filtro (A) por edicion."""
    l21, l22, l23, _ = R.load_data()
    raw = pd.concat([l21, l22, l23])
    with contextlib.redirect_stdout(io.StringIO()):
        d = R.all_laps()
    wet = wet_editions(raw)
    d = d[[(y, g) not in wet for y, g in zip(d['Year'], d['GP'])]]
    d = add_stint(d, raw)
    d['GP'] = d['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')
    return d


def laps_2024_A():
    _, _, _, raw = R.load_data()
    wet = wet_editions(raw)
    d = R.filter_laps(raw[[(y, g) not in wet for y, g in zip(raw['Year'], raw['GP'])]]).copy()
    d['DRS'] = ((d['Interval_front'] < 1.0) & (d['LapNumber'] > 2)).astype(int)
    d['Interval_front'] = np.exp(-d['Interval_front']); d['Interval_behind'] = np.exp(-d['Interval_behind'])
    d['LapNumber_2'] = d['LapNumber'] ** 2
    d['FirstLap_pos'] = d['Position'] * (d['LapNumber'] == 1.0)
    med = d.groupby('GP')['LapTime'].transform('median')
    d = d[~((d['LapTime'] >= med + 15) & ~((d['PitIn'] == 1) | (d['PitOut'] == 1) | (d['LapNumber'] == 1)))]
    return add_stint(d, raw)


def fit_gp(d, gp):
    with contextlib.redirect_stdout(io.StringIO()):
        m, dd = R.filtrar_outliers_iqr(d, gp)
    P, V = m.params, m.cov_params()
    ref = P.index[P.index.str.contains('Compound_Detail, Treatment')][0].split("'")[1]
    comps = sorted(dd.Compound_Detail.unique())
    names, kinds = [], []
    for c in comps:
        if c != ref:
            names.append(f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"); kinds.append(('off', c))
    for c in comps:
        names.append(f"TyreLife:C(Compound_Detail)[{c}]"); kinds.append(('sl', c))
    stints = {c: s.groupby(['Year', 'Driver', 'stint']).ngroups for c, s in dd.groupby('Compound_Detail')}
    return dict(ref=ref, comps=comps, kinds=kinds, b=P[names].values, V=V.loc[names, names].values,
                laps=dd.Compound_Detail.value_counts().to_dict(), stints=stints)


# --------------------------------------------------------------------------- modelo conjunto
ci = lambda c: int(c[1:])
STEPS = 5  # C0->C1, ..., C4->C5
COMPS_FREE = ['C0', 'C1', 'C2', 'C4', 'C5']


def fit_pooled(H, GPS, kind, tau_t, tau_e, mono=False, tie45=False):
    """Devuelve {(gp, compuesto): valor} para todas las celdas con historia o usadas en 2024."""
    cells = sorted({(g, c) for g in GPS for c in set(H[g]['comps']) | set(MAP[g])})
    cidx = {k: i for i, k in enumerate(cells)}; G = {g: i for i, g in enumerate(GPS)}
    nG, nC = len(GPS), len(cells)
    use_e = tau_e > 0
    B = {}
    for g in GPS:
        r = H[g]; idx = [i for i, k in enumerate(r['kinds']) if k[0] == kind]
        B[g] = ([r['kinds'][i][1] for i in idx], r['b'][idx],
                np.linalg.inv(np.linalg.cholesky(r['V'][np.ix_(idx, idx)])))

    def pattern(p):
        if mono:
            st = np.exp(p[:STEPS]).copy()
            if tie45: st[4] = st[3]
            cum = np.concatenate([[0], np.cumsum(st)])
            sgn = -1 if kind == 'off' else 1
            return {f'C{c}': sgn * (cum[c] - cum[3]) for c in range(6)}
        a = dict(zip(COMPS_FREE, p[:STEPS])); a['C3'] = 0.0
        return a

    def mean(p, a, g, c):
        lev, tilt, e = p[STEPS:STEPS + nG], p[STEPS + nG:STEPS + 2 * nG], p[STEPS + 2 * nG:]
        t = e[cidx[(g, c)]] if use_e else 0.0
        if mono:
            lam = np.exp(tilt[G[g]])
            return lam * a[c] + t if kind == 'off' else np.exp(lev[G[g]] + lam * a[c]) + t
        if kind == 'off':
            return a[c] + tilt[G[g]] * (ci(c) - 3) + t
        return np.exp(lev[G[g]] + a[c] + tilt[G[g]] * (ci(c) - 3)) + t

    def resid(p):
        a = pattern(p); out = []
        for g in GPS:
            comps, y, Li = B[g]
            if not comps: continue
            ref = H[g]['ref']
            pred = np.array([mean(p, a, g, c) - (mean(p, a, g, ref) if kind == 'off' else 0) for c in comps])
            out.append(Li @ (y - pred))
        tilt = p[STEPS + nG:STEPS + 2 * nG]
        out.append(tilt / tau_t)
        if kind == 'off': out.append(p[STEPS:STEPS + nG] * 1e-3)   # nivel no identificado en fijos
        out.append(p[STEPS + 2 * nG:] / tau_e if use_e else p[STEPS + 2 * nG:])
        return np.concatenate(out)

    p0 = np.zeros(STEPS + 2 * nG + nC)
    if mono: p0[:STEPS] = np.log(0.2)
    if kind == 'sl': p0[STEPS:STEPS + nG] = np.log(0.06)
    sol = least_squares(resid, p0, method='trf', x_scale='jac', max_nfev=20000)
    a = pattern(sol.x)
    return {k: float(mean(sol.x, a, *k)) for k in cells}, a


OPTIONS = {
    'opcion2': dict(desc='conjunto libre + efecto de celda (0.1 s / 0.01 s/vuelta)',
                    off=dict(tau_t=0.15, tau_e=0.1), sl=dict(tau_t=0.15, tau_e=0.01), floor_slope=0.0),
    'opcion3': dict(desc='conjunto monotono estructural, paso C4->C5 = paso C3->C4',
                    off=dict(tau_t=0.3, tau_e=0, mono=True, tie45=True),
                    sl=dict(tau_t=0.3, tau_e=0, mono=True, tie45=True), floor_slope=None),
}


def current_sim():
    fx = pd.read_excel(config.DATA_DIR / 'completed_coefs_V2_filled.xlsx', sheet_name='fixed', index_col=0)
    vr = pd.read_excel(config.DATA_DIR / 'completed_coefs_V2_filled.xlsx', sheet_name='variable', index_col=0)
    o = {(g, c): fx.at[g, c] for g in fx.index for c in MAP.get(g, []) if c in fx and pd.notna(fx.at[g, c])}
    s = {(g, c): vr.at[g, c] for g in vr.index for c in MAP.get(g, []) if c in vr and pd.notna(vr.at[g, c])}
    return o, s


# --------------------------------------------------------------------------- validacion
def truth_table(Y, GPS):
    rows = []
    for g in GPS:
        if g not in Y: continue
        r = Y[g]; comps = [c for c in MAP[g] if c in r['comps'] and r['laps'].get(c, 0) >= 30]
        on = lambda c: None if c == r['ref'] else r['kinds'].index(('off', c))
        sn = lambda c: r['kinds'].index(('sl', c))
        for c in comps:
            rows.append(dict(GP=g, kind='slope', pair=c, age=np.nan, t=r['b'][sn(c)],
                             se=np.sqrt(r['V'][sn(c), sn(c)])))
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                h, s_ = comps[i], comps[j]
                for age in (0, 10, 20):
                    w = np.zeros(len(r['b']))
                    if on(s_) is not None: w[on(s_)] += 1
                    if on(h) is not None: w[on(h)] -= 1
                    w[sn(s_)] += age; w[sn(h)] -= age
                    rows.append(dict(GP=g, kind='delta', pair=f'{s_}-{h}', age=age, t=w @ r['b'],
                                     se=np.sqrt(w @ r['V'] @ w)))
    return pd.DataFrame(rows)


def predict(T, o, s):
    out = []
    for _, t in T.iterrows():
        if t.kind == 'slope':
            out.append(s.get((t.GP, t.pair), np.nan)); continue
        sc, hc = t.pair.split('-')
        out.append(o.get((t.GP, sc), np.nan) - o.get((t.GP, hc), np.nan)
                   + t.age * (s.get((t.GP, sc), np.nan) - s.get((t.GP, hc), np.nan)))
    return np.array(out)


def main():
    dh, d24 = hist_laps_A(), laps_2024_A()
    H = {g: fit_gp(dh, g) for g in MAP if g in set(dh.GP)}
    Y = {}
    for g in MAP:
        if g in set(d24.GP):
            try: Y[g] = fit_gp(d24, g)
            except Exception as e: print('2024 sin regresion:', g, e)
    GPS = [g for g in MAP if g in H]
    print(f'GP con historia 2021-23 (filtro A): {len(GPS)}; con regresion 2024: {len(Y)}')

    opts = {'actual': current_sim()}
    for name, cfg in OPTIONS.items():
        o, pat_o = fit_pooled(H, GPS, 'off', **cfg['off'])
        s, pat_s = fit_pooled(H, GPS, 'sl', **cfg['sl'])
        if cfg['floor_slope'] is not None:
            s = {k: max(v, cfg['floor_slope']) for k, v in s.items()}
        opts[name] = (o, s)
        print(f'{name}: patron fijo comun', {k: round(v, 3) for k, v in sorted(pat_o.items())},
              '| pendiente relativa a C3', {k: round(float(np.exp(v)), 3) for k, v in sorted(pat_s.items())})

    # coeficientes para el simulador: fijo relativo al compuesto de referencia de la Ec.1 (filtro A)
    for name in OPTIONS:
        o, s = opts[name]
        rows = [dict(GP=g, ref=H[g]['ref'], compound=c,
                     offset_vs_ref=o[(g, c)] - o[(g, H[g]['ref'])], slope=s[(g, c)],
                     stints_2021_23=H[g]['stints'].get(c, 0))
                for g in GPS for c in MAP[g]]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'coefs_{name}.csv'), index=False)

    # validacion
    T = truth_table(Y, GPS)
    res = []
    for name, (o, s) in opts.items():
        P = T.assign(p=predict(T, o, s), opcion=name); res.append(P)
    V = pd.concat(res); V['err'] = V.p - V.t
    V.to_csv(os.path.join(OUT, 'validation_cells.csv'), index=False)
    summ = []
    for lab, sub in [('todos', V), ('8 GP paper', V[V.GP.isin(PAPER)])]:
        for name, x in sub.groupby('opcion', sort=False):
            dl = x[x.kind == 'delta'].dropna(subset=['p']); sl = x[x.kind == 'slope'].dropna(subset=['p'])
            summ.append(dict(conjunto=lab, opcion=name, n_delta=len(dl), MAE_delta=dl.err.abs().mean(),
                             MAE_delta_w=np.average(dl.err.abs(), weights=1 / dl.se ** 2),
                             n_slope=len(sl), MAE_slope=sl.err.abs().mean(), bias_slope=sl.err.mean()))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, 'validation_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(S.round(4).to_string(index=False))

    # tabla legible por GP del paper (fijo relativo al medio | pendiente)
    rows = []
    for g in PAPER:
        m = MAP[g][1]
        r = Y[g]; o24 = {(g, r['ref']): 0.0}; s24 = {}
        for (k, c), b in zip(r['kinds'], r['b']): (o24 if k == 'off' else s24)[(g, c)] = b
        for name, (o, s) in [('2024 (vara)', (o24, s24))] + list(opts.items()):
            for c in MAP[g]:
                rows.append(dict(GP=g.replace(' Grand Prix', ''), fuente=name, compound=c,
                                 fijo_vs_medio=o.get((g, c), np.nan) - o.get((g, m), np.nan),
                                 pendiente=s.get((g, c), np.nan)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'paper_gps_table.csv'), index=False)


if __name__ == '__main__':
    main()
