"""2026-10-10 — Ec. 2 (correccion de coeficientes de piloto): comparacion de 4 especificaciones.

Correr desde la raiz del repo: python results/driver_coef_study_20261010/eq2_variants_20261010.py
Entradas: race_coefs.csv (race_coefs_20261010.py) y data/session_bestlaps_2021_2024_20261010.csv.
Salida: results/driver_coef_study_20261010/eq2_*.csv

z = coeficiente de piloto de la carrera (Ec. 1 ajustada solo con esa carrera; s/vuelta vs VER).
Predictores: coeficientes de las 3 carreras previas del piloto (cronologicas, cruzando temporadas),
mejor vuelta en FP1-FP3 y en la clasificacion del mismo fin de semana (disponibles antes de la carrera).
Variantes:
  bruto     : como el paper: Prev1-3 + BestFP + BestQualy en segundos brutos + Length
              (proxy: pole del fin de semana, en s; no hay km en los datos)
  dif_VER   : Prev1-3 + (BestFP - BestFP_VER) + (BestQualy - BestQualy_VER)
  dif_med   : Prev1-3 + (BestFP - mediana de la sesion) + (BestQualy - mediana de la sesion)
  porcentual: todo dividido por la pole del fin de semana correspondiente (z, Prev_i con la pole de su
              propia carrera; FP y Q como % sobre VER); la prediccion se multiplica por la pole del GP objetivo
Validacion: ventana expansiva (solo carreras ANTERIORES al GP objetivo, cruzando anios; >= 30 filas),
prediccion de cada carrera 2023 y 2024. Lineas base: sin_correccion_prev1 (= Prev1, como la Fig. 4 del
codigo) y sin_correccion_hist (coeficiente de la Ec. 1 2021..anio-1 del GP, lo que usa el simulador).
Muestra: pilotos != VER con >= 15 vueltas limpias en la carrera objetivo; sesiones con >10 % WET/INT -> NaN.
"""
import io, os, sys, contextlib
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import statsmodels.api as sm
import models.regressions as R

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_LAPS, MIN_TRAIN = 15, 30
PAPER = ['Bahrain', 'Saudi Arabian', 'Miami', 'Hungarian', 'Belgian', 'Dutch', 'Singapore', 'United States']
NAME = {'Styrian Grand Prix': 'Styrian Grand Prix'}  # ajustar si FastF1 y las sesiones difieren


def load():
    rc = pd.read_csv(os.path.join(HERE, 'race_coefs.csv'))
    bl = pd.read_csv(os.path.join(ROOT, 'data', 'session_bestlaps_2021_2024_20261010.csv'))
    bl['GP'] = bl.GP.replace(NAME)
    bl.loc[bl.wet_share > 0.10, 'BestLapTime'] = np.nan
    rnd = bl.groupby(['Year', 'GP']).Round.first()
    miss = sorted(set(map(tuple, rc[['Year', 'GP']].drop_duplicates().values)) - set(rnd.index))
    if miss: print('carreras sin sesiones FastF1 (nombre distinto?):', miss)
    fp = bl[bl.Session.str.startswith('Practice')].groupby(['Year', 'GP', 'Driver']).BestLapTime.min().rename('FP')
    q = bl[bl.Session == 'Qualifying'].groupby(['Year', 'GP', 'Driver']).BestLapTime.min().rename('Q')
    s = pd.concat([fp, q], axis=1).reset_index()
    for c in ('FP', 'Q'):
        g = s.groupby(['Year', 'GP'])[c]
        s[c + '_med'] = s[c] - g.transform('median')
        ver = s[s.Driver == 'VER'].set_index(['Year', 'GP'])[c]
        s[c + '_ver'] = s[c] - s.set_index(['Year', 'GP']).index.map(ver).values
        s[c + '_pct'] = s[c + '_ver'] / s.set_index(['Year', 'GP']).index.map(ver).values
    pole = s.groupby(['Year', 'GP']).Q.min().fillna(s.groupby(['Year', 'GP']).FP.min()).rename('pole')
    d = rc.merge(s, on=['Year', 'GP', 'Driver'], how='left').merge(pole.reset_index(), on=['Year', 'GP'], how='left')
    d['Round'] = d.set_index(['Year', 'GP']).index.map(rnd).values
    d = d.dropna(subset=['Round'])
    d['t'] = d.Year * 100 + d.Round
    d['z_pct'] = d.coef / d.pole
    # carreras previas del piloto (cronologicas; solo carreras con coeficiente fiable)
    d = d.sort_values(['Driver', 't'])
    out = []
    for drv, g in d.groupby('Driver'):
        g = g.copy(); v = g[g.n_laps >= MIN_LAPS]
        for k in (1, 2, 3):
            for col, src in ((f'P{k}', 'coef'), (f'P{k}_pct', 'z_pct')):
                prev = [v[v.t < t][src].iloc[-k] if (v.t < t).sum() >= k else np.nan for t in g.t]
                g[col] = prev
        out.append(g)
    return pd.concat(out)


def hist_coefs(d):
    """Coeficiente de piloto del simulador: Ec. 1 con 2021..anio-1 (filtro A) para cada GP objetivo."""
    raw = pd.concat(R.load_data(), ignore_index=True)
    share = raw.groupby(['Year', 'GP'])['Compound'].apply(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
    wet = set(share[share > 0.10].index)
    base = R.filter_laps(raw[[(y, g) not in wet for y, g in zip(raw.Year, raw.GP)]]).copy()
    base['DRS'] = ((base['Interval_front'] < 1.0) & (base['LapNumber'] > 2)).astype(int)
    base['Interval_front'] = np.exp(-base['Interval_front']); base['Interval_behind'] = np.exp(-base['Interval_behind'])
    base['LapNumber_2'] = base['LapNumber'] ** 2
    base['FirstLap_pos'] = base['Position'] * (base['LapNumber'] == 1.0)
    med = base.groupby(['Year', 'GP'])['LapTime'].transform('median')
    base = base[~((base['LapTime'] >= med + 15) & ~((base['PitIn'] == 1) | (base['PitOut'] == 1) | (base['LapNumber'] == 1)))]
    rows = []
    pre = "C(Driver, Treatment(reference='VER'))[T."
    for (y, g) in d[d.Year >= 2023][['Year', 'GP']].drop_duplicates().itertuples(index=False):
        h = base[(base.GP == g) & (base.Year < y)]
        if h.empty or 'VER' not in set(h.Driver): continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m, _ = R.filtrar_outliers_iqr(h, g)
        except Exception:
            continue
        for k in m.params.index:
            if k.startswith(pre):
                rows.append(dict(Year=y, GP=g, Driver=k[len(pre):-1], hist=float(m.params[k])))
    return pd.DataFrame(rows)


SPECS = {
    'bruto': ['P1', 'P2', 'P3', 'FP', 'Q', 'pole'],
    'dif_VER': ['P1', 'P2', 'P3', 'FP_ver', 'Q_ver'],
    'dif_med': ['P1', 'P2', 'P3', 'FP_med', 'Q_med'],
    'porcentual': ['P1_pct', 'P2_pct', 'P3_pct', 'FP_pct', 'Q_pct'],
}


def main():
    d = load()
    d = d.merge(hist_coefs(d), on=['Year', 'GP', 'Driver'], how='left')
    d.to_csv(os.path.join(HERE, 'eq2_features.csv'), index=False)
    sample = d[(d.Driver != 'VER') & (d.n_laps >= MIN_LAPS)]
    preds = []
    for (y, g, t), tg in sample[sample.Year >= 2023].groupby(['Year', 'GP', 't']):
        tr_all = sample[sample.t < t]
        r = tg[['Year', 'GP', 'Driver', 'coef', 'pole', 'P1', 'hist']].copy()
        for name, cols in SPECS.items():
            ycol = 'z_pct' if name == 'porcentual' else 'coef'
            tr = tr_all.dropna(subset=cols + [ycol])
            r[name] = np.nan
            if len(tr) < MIN_TRAIN: continue
            m = sm.OLS(tr[ycol], sm.add_constant(tr[cols])).fit()
            ok = tg[cols].notna().all(axis=1)
            if ok.any():
                p = m.predict(sm.add_constant(tg.loc[ok, cols], has_constant='add'))
                r.loc[ok, name] = p * (tg.loc[ok, 'pole'] if name == 'porcentual' else 1)
        preds.append(r)
    P = pd.concat(preds).rename(columns={'P1': 'sin_correccion_prev1', 'hist': 'sin_correccion_hist'})
    P.to_csv(os.path.join(HERE, 'eq2_predictions.csv'), index=False)
    meth = list(SPECS) + ['sin_correccion_prev1', 'sin_correccion_hist']
    # comparacion sobre filas donde TODOS los metodos predicen (misma muestra)
    C = P.dropna(subset=meth)
    rows = []
    for lab, sub in [('2023', C[C.Year == 2023]), ('2024', C[C.Year == 2024]),
                     ('2024 8 GP paper', C[(C.Year == 2024) & C.GP.str.replace(' Grand Prix', '').isin(PAPER)])]:
        rows.append(dict(muestra=lab, n=len(sub), **{m: ((sub[m] - sub.coef) ** 2).mean() for m in meth}))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, 'eq2_mse_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print('MSE (s/vuelta)^2, misma muestra para todos los metodos:'); print(S.round(4).to_string(index=False))
    cov = P.groupby('Year')[meth].apply(lambda x: x.notna().mean()).round(2)
    print('cobertura (fraccion de filas con prediccion):'); print(cov.to_string())
    per = P[P.Year == 2024].groupby('GP').apply(lambda x: pd.Series({m: ((x[m] - x.coef) ** 2).mean() for m in meth}))
    per.to_csv(os.path.join(HERE, 'eq2_mse_by_gp_2024.csv')); print(per.round(3).to_string())
    for name, cols in SPECS.items():   # coeficientes con todo 2021-24 (ilustrativo)
        ycol = 'z_pct' if name == 'porcentual' else 'coef'
        tr = sample.dropna(subset=cols + [ycol])
        m = sm.OLS(tr[ycol], sm.add_constant(tr[cols])).fit()
        print(f'\n{name} (n={int(m.nobs)}, R2={m.rsquared:.3f})'); print(pd.concat([m.params, m.bse], axis=1, keys=['coef', 'se']).round(4).to_string())


if __name__ == '__main__':
    main()
