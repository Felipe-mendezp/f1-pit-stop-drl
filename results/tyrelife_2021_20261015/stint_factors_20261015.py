"""2026-10-15: factor de reescalamiento de TyreLife 2021 (equivalente 2022-23), revision de tires_2021.py.

Codigo vigente: src/models/rival_models.py (y los dos logit de adelantamiento) multiplican TyreLife 2021 por
TYRELIFE_ADJUST = {C0 .65, C1 1.0, C2 .90, C3 .98, C4 .92, C5 .92} (origen: Codigos/respaldo/.../tires_2021.py).

Metodos:
  M0  replica de tires_2021.py: media de km del stint, 2022-24 / 2021, mismos GP de 2021, stints >= 8 vueltas y ultima
      vuelta con TrackStatus == 1 (incluye stints finales cortados por la bandera a cuadros).
  M1  igual que M0 pero solo stints terminados en parada (excluye el ultimo stint, censurado por el fin de carrera),
      parada en verde (vuelta de entrada y la anterior sin SC/VSC/roja), largo medido como TyreLife en la vuelta de
      entrada (es lo que se reescala; incluye vueltas previas del juego usado), 2022-23 (los anios de ajuste).
  M2  M1 pareado: log(TyreLife al parar) ~ C(GP):C(compuesto) [celdas] + es2021:C(compuesto), solo celdas GP-compuesto
      con datos en 2021 y en 2022-23. Factor = exp(-coef 2021). Tambien un factor comun.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

D = os.path.join(ROOT, 'data')
df = pd.concat([pd.read_csv(os.path.join(D, f'session_{y}_V2.csv')).assign(Year=y) for y in range(2021, 2025)],
               ignore_index=True)
df = df[~df.Compound.isin(['WET', 'INTERMEDIATE'])].sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
ts = df.TrackStatus.astype(str).str.replace(r'\.0$', '', regex=True)
df['neutral'] = ts.str.contains('[4567]')
g = df.groupby(['Year', 'GP', 'Driver'])
df['stint'] = g.PitIn.transform(lambda s: s.shift(fill_value=0).cumsum())     # la vuelta de entrada cierra su stint
df['neutral_prev'] = g.neutral.shift(fill_value=False)
st = (df.groupby(['Year', 'GP', 'Driver', 'stint'])
        .agg(comp=('Compound_Detail', 'first'), ncomp=('Compound_Detail', 'nunique'), laps=('LapNumber', 'count'),
             tl_end=('TyreLife', 'last'), tl_start=('TyreLife', 'first'), pit=('PitIn', 'last'),
             last_status=('TrackStatus', 'last'), neutral_end=('neutral', 'last'), neutral_prev=('neutral_prev', 'last'))
        .reset_index())
st = st[st.ncomp == 1]
gp21 = df[df.Year == 2021].GP.unique()


def m0(comp):
    s = st[(st.comp == comp) & st.GP.isin(gp21) & (st.laps >= 8) & (st.last_status.astype(float) == 1)]
    a, b = s[s.Year == 2021].laps.mean(), s[s.Year >= 2022].laps.mean()        # (km ~ vueltas: mismos GP)
    return b / a, (s.Year == 2021).sum()


ok = (st.pit == 1) & ~st.neutral_end & ~st.neutral_prev & (st.laps >= 8) & st.Year.le(2023)
s1 = st[ok].copy()
rows = []
for c in ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']:
    f0, n0 = m0(c)
    a = s1[(s1.comp == c) & (s1.Year == 2021) & s1.GP.isin(gp21)]
    b = s1[(s1.comp == c) & (s1.Year >= 2022) & s1.GP.isin(gp21)]
    rows.append(dict(comp=c, M0=f0, n21_M0=n0, M1=b.tl_end.mean() / a.tl_end.mean() if len(a) else np.nan,
                     n21=len(a), n2223=len(b)))
R = pd.DataFrame(rows).set_index('comp')

# M2 pareado
s1['cell'] = s1.GP + '|' + s1.comp
both = s1.groupby('cell').Year.agg(lambda y: (y == 2021).any() and (y >= 2022).any())
s2 = s1[s1.cell.map(both)].copy()
s2['y21'] = (s2.Year == 2021).astype(int)
s2['ltl'] = np.log(s2.tl_end)
fit = smf.ols('ltl ~ C(cell) + y21:C(comp)', data=s2).fit(cov_type='cluster', cov_kwds={'groups': s2.GP})
for c in R.index:
    k = f'y21:C(comp)[{c}]'
    if k in fit.params:
        R.loc[c, 'M2'] = np.exp(-fit.params[k])
        lo, hi = fit.conf_int().loc[k]
        R.loc[c, 'M2_ic95'] = f'{np.exp(-hi):.2f}-{np.exp(-lo):.2f}'
        R.loc[c, 'celdas_M2'] = s2[(s2.comp == c)].cell.nunique()
        R.loc[c, 'n21_M2'] = ((s2.comp == c) & (s2.y21 == 1)).sum()
fc = smf.ols('ltl ~ C(cell) + y21', data=s2).fit(cov_type='cluster', cov_kwds={'groups': s2.GP})
lo, hi = fc.conf_int().loc['y21']
R['vigente'] = pd.Series({'C0': .65, 'C1': 1.0, 'C2': .90, 'C3': .98, 'C4': .92, 'C5': .92})
pd.set_option('display.width', 200)
print(R.round(3).to_string())
print(f'\nM2 factor comun: {np.exp(-fc.params["y21"]):.3f} (IC95 {np.exp(-hi):.3f}-{np.exp(-lo):.3f}), '
      f'{s2.cell.nunique()} celdas, {s2.y21.sum()} stints 2021, {(1 - s2.y21).sum()} stints 2022-23')
# celdas por GP del C0 (compuesto con factor mas extremo)
print('\nC0 por GP (TyreLife medio al parar, stints en verde terminados en parada):')
print(s1[s1.comp == 'C0'].groupby(['GP', 'Year']).tl_end.agg(['mean', 'size']).unstack().round(1).to_string())
R.to_csv(os.path.join(HERE, 'stint_factors_20261015.csv'))

# ---- comparacion por anio frente a 2022-23 (mismo criterio M1/M2, celdas GP-compuesto pareadas), incluye 2024
s4 = st[(st.pit == 1) & ~st.neutral_end & ~st.neutral_prev & (st.laps >= 8)].copy()
s4['cell'] = s4.GP + '|' + s4.comp
s4['ltl'] = np.log(s4.tl_end)
for y in [2021, 2024]:
    s = s4[s4.Year.isin([y, 2022, 2023])]
    both = s.groupby('cell').Year.agg(lambda t: (t == y).any() and (t != y).any())
    s = s[s.cell.map(both)].assign(yy=lambda d: (d.Year == y).astype(int))
    f = smf.ols('ltl ~ C(cell) + yy', data=s).fit(cov_type='cluster', cov_kwds={'groups': s.GP})
    lo, hi = f.conf_int().loc['yy']
    print(f'{y} frente a 2022-23: TyreLife al parar x{np.exp(f.params["yy"]):.3f} (IC95 {np.exp(lo):.3f}-{np.exp(hi):.3f}); '
          f'factor para llevarlo a 2022-23: {np.exp(-f.params["yy"]):.3f}; {s.cell.nunique()} celdas, {s.yy.sum()} stints')
print('Media simple de TyreLife al parar por anio:', s4.groupby('Year').tl_end.mean().round(2).to_dict())
