"""
2026-10-07 — Modelo de tiempos de vuelta bajo SC/VSC "v4" (reemplaza la regresión YF 2024 del autor).

Estudio y justificación: results/yf_pitloss_study_20261007/ (README y scripts 01-11).

Datos 2021-2023 (data/session_202{1,2,3}_V2.csv), filtro A (carreras con <=10% de vueltas WET/INT;
neumáticos de lluvia excluidos), sin la vuelta 1. Solo vueltas que el piloto corrió 100% bajo un
estado (TrackStatus de la vuelta = '1', '4' o '6'):

  LapTime/Lref_g = rho_VSC*VSC + rho_SCl*SC_libre + rho_SCf*SC_fila + u_g*SC            (ritmo)
                 + k_in*PitIn*Gin_g/Lref_g + k_out*PitOut*Gout_g/Lref_g                   (pit)
                 + v_(carrera,vuelta) + e

  Lref_g        vuelta verde de referencia del GP (mediana de vueltas limpias sin pit)
  Gin_g, Gout_g pérdida verde de la in-lap y la out-lap del GP: exceso sobre la mediana de los autos
                sin pit de la misma vuelta (>=3 autos con vuelta verde pura)
  SC_libre      2a vuelta real del evento (antes de que la fila se forme) = 1a vuelta SC del simulador
  SC_fila       vueltas >=3 del evento (autos alineados)
  u_g           efecto GP en el ritmo SC; v_(carrera,vuelta) ruido común de cada vuelta
Se excluyen la vuelta real del despliegue (cada auto la corre parcialmente bajo SC) y las paradas en
vueltas SC_fila (su tiempo mezcla el reagrupamiento, que el simulador modela aparte). SC y VSC
comparten k_in, k_out (mismo ritmo antes de juntarse; ver estudio, scripts 07-08).

Coeficientes del simulador (modelo YF_TERMS de train_yf_models.py, sin intercepto):
  PitIn = k_in*Gin_g   PitOut = k_out*Gout_g   VSC = rho_VSC*Lref_g   VSC_firstlap = 0
  SC = (rho_SCf + u_g)*Lref_g   SC_firstlap = (rho_SCl - rho_SCf)*Lref_g
  (con --sc-bunching el entorno usa el SC plano también en la 1a vuelta; ver _simulate_lap_times)

Salida (no sobrescribe nada existente): data/yf_model_v4_20261007.csv (una fila por GP) y
data/yf_model_v4_20261007_fit.txt (resumen de la regresión).

    python scripts/fit_yf_model_v4_20261007.py
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT_CSV = os.path.join(DATA, 'yf_model_v4_20261007.csv')
OUT_TXT = os.path.join(DATA, 'yf_model_v4_20261007_fit.txt')
YEARS = (2021, 2022, 2023)
WET_MAX = 0.10


def lap_state(ts):
    s = str(ts).split('.')[0]
    if '5' in s:
        return 'RED'
    if '4' in s:
        return 'SC'
    if '6' in s:
        return 'VSC'
    return 'G'


def load_laps():
    d = pd.concat([pd.read_csv(os.path.join(DATA, f'session_{y}_V2.csv')) for y in YEARS],
                  ignore_index=True)
    d['wet'] = d['Compound'].isin(['WET', 'INTERMEDIATE'])
    wet_share = d.groupby(['Year', 'GP'])['wet'].transform('mean')
    d = d[(wet_share <= WET_MAX) & ~d['wet']].copy()
    d['ts'] = d['TrackStatus'].astype(str).str.split('.').str[0]
    d['pure'] = np.select([d.ts == '1', d.ts == '4', d.ts == '6'], ['G', 'SC', 'VSC'], 'mix')
    # Índice de vuelta dentro del evento SC/VSC (estado modal de la carrera-vuelta)
    k = ['Year', 'GP', 'LapNumber']
    lap = (d.assign(st=d['TrackStatus'].map(lap_state))
           .groupby(k)['st'].agg(lambda s: s.mode().iat[0]).rename('lap_st')
           .reset_index().sort_values(k))
    idx, prev, cnt, last = [], None, 0, 'G'
    for r in lap.itertuples():
        if (r.Year, r.GP) != prev:
            prev, cnt, last = (r.Year, r.GP), 0, 'G'
        cnt = (cnt + 1 if last == r.lap_st else 1) if r.lap_st in ('SC', 'VSC') else 0
        last = r.lap_st
        idx.append(cnt)
    lap['ev_idx'] = idx
    d = d.merge(lap[k + ['ev_idx']], on=k)
    d = d[(d.LapNumber > 1) & d.LapTime.notna()]
    clean = d[(d.pure == 'G') & (d.PitIn == 0) & (d.PitOut == 0)]
    d = d.join(clean.groupby(['Year', 'GP'])['LapTime'].median().rename('Lref'), on=['Year', 'GP'])
    return d


def green_components(d):
    """Pérdida verde por GP (in-lap, out-lap): exceso sobre la mediana de >=3 autos sin pit de la
    misma vuelta, con vuelta verde pura."""
    k = ['Year', 'GP', 'LapNumber']
    nop = d[(d.PitIn == 0) & (d.PitOut == 0) & (d.pure == 'G')]
    ref = nop.groupby(k)['LapTime'].agg(['median', 'size'])
    g = d[d.pure == 'G'].join(ref, on=k)
    g = g[g['size'] >= 3]
    g['exc'] = g.LapTime - g['median']
    g = g[g.exc.abs() < 60]
    return pd.DataFrame({'Gin': g[g.PitIn == 1].groupby('GP')['exc'].median(),
                         'Gout': g[g.PitOut == 1].groupby('GP')['exc'].median(),
                         'n_green_in': g[g.PitIn == 1].groupby('GP').size(),
                         'n_green_out': g[g.PitOut == 1].groupby('GP').size()})


def main():
    for p in (OUT_CSV, OUT_TXT):
        if os.path.exists(p):
            sys.exit(f'{p} ya existe; no se sobrescribe. Bórralo tú si quieres regenerarlo.')
    d = load_laps()
    G = green_components(d)
    Lg = d.groupby('GP')['Lref'].median().rename('Lref')

    y = d[d.pure.isin(['SC', 'VSC'])].copy()
    y['VSC'] = (y.pure == 'VSC').astype(int)
    y['SCl'] = ((y.pure == 'SC') & (y.ev_idx == 2)).astype(int)
    y['SCf'] = ((y.pure == 'SC') & (y.ev_idx >= 3)).astype(int)
    y = y[(y.VSC + y.SCl + y.SCf) == 1]
    y = y[~(((y.PitIn + y.PitOut) > 0) & (y.SCf == 1))]
    y = y.join(G[['Gin', 'Gout']], on='GP').dropna(subset=['Gin', 'Gout'])
    y['ratio'] = y.LapTime / y.Lref
    y = y[(y.ratio > 0.9) & (y.ratio < 2.2)]
    y['PI'] = y.PitIn * y.Gin / y.Lref
    y['PO'] = y.PitOut * y.Gout / y.Lref
    y['SC'] = y.SCl + y.SCf
    y['rl'] = y.Year.astype(str) + '|' + y.GP + '|' + y.LapNumber.astype(int).astype(str)

    m = smf.mixedlm('ratio ~ 0 + VSC + SCl + SCf + PI + PO', y, groups=y['GP'], re_formula='~0+SC',
                    vc_formula={'rl': '0+C(rl)'}).fit(method='lbfgs')
    fe = m.fe_params
    u = {gp: float(r.iloc[0]) for gp, r in m.random_effects.items()}

    rows = []
    for gp in sorted(set(G.dropna().index) & set(Lg.index)):
        L, gi, go, ug = Lg[gp], G.loc[gp, 'Gin'], G.loc[gp, 'Gout'], u.get(gp, 0.0)
        rows.append({
            'GP': gp, 'Lref': L, 'Gin': gi, 'Gout': go,
            'n_green_in': int(G.loc[gp, 'n_green_in']), 'n_green_out': int(G.loc[gp, 'n_green_out']),
            'n_sc_laps': int(y[(y.GP == gp) & (y.SC == 1)].shape[0]),
            'rho_VSC': fe['VSC'], 'rho_SC_libre': fe['SCl'] + ug, 'rho_SC_fila': fe['SCf'] + ug,
            'u_SC': ug, 'k_in': fe['PI'], 'k_out': fe['PO'],
            # coeficientes del simulador (YF_TERMS)
            'PitIn': fe['PI'] * gi, 'PitOut': fe['PO'] * go,
            'SC': (fe['SCf'] + ug) * L, 'SC_firstlap': (fe['SCl'] - fe['SCf']) * L,
            'VSC': fe['VSC'] * L, 'VSC_firstlap': 0.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format='%.6f')

    summ = [__doc__.split('Coeficientes del simulador')[0].strip(), '',
            f'Muestra: {len(y)} vueltas, {y[["Year", "GP"]].drop_duplicates().shape[0]} carreras; '
            f'VSC {int(y.VSC.sum())}, SC libre {int(y.SCl.sum())}, SC fila {int(y.SCf.sum())}, '
            f'PitIn {int(y.PitIn.sum())}, PitOut {int(y.PitOut.sum())}', '',
            str(m.summary().tables[1].iloc[:7].round(4)), '',
            f'sd efecto GP en SC: {np.sqrt(m.cov_re.iloc[0, 0]):.4f}', f'converged: {m.converged}']
    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(summ) + '\n')
    print('\n'.join(summ[2:]))
    print(f'\n{OUT_CSV}: {len(out)} GPs')


if __name__ == '__main__':
    main()
