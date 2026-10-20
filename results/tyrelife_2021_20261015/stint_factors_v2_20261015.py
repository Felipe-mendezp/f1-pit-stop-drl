"""2026-10-15 (v2): factor conductual de TyreLife 2021 incluyendo el ultimo stint (criterio del autor).

Idea: la vuelta de cada parada se elige sabiendo cuantas vueltas quedan, asi que el ultimo stint tambien informa cuanto
aguanta un neumatico. Se excluye lo que no responde al desgaste:
  - stints terminados en parada bajo SC/VSC/roja (vuelta de entrada o la anterior neutralizada);
  - ultimos stints que empezaron con una parada bajo SC/VSC/roja (su largo lo decide la neutralizacion);
  - ultimos stints cortos (< MINF vueltas: paradas tardias por la vuelta rapida u oportunistas);
  - pilotos que no terminan (ultima vuelta < vueltas de la carrera - 2) y stints con mas de un compuesto.
Stints intermedios: igual que M2 de stint_factors_20261015.py (>= 8 vueltas, terminados en parada en verde).
Largo = TyreLife en la ultima vuelta del stint (vuelta de entrada, o la ultima vuelta de la carrera).

Metodo (pareado como M2): log(TyreLife final) ~ C(GP:compuesto) + y21 [+ final], celdas con 2021 y 2022-23.
Variantes: V1 = M2 (sin ultimos stints), V2 = con ultimos stints, V3 = V2 + control por ultimo stint,
V4 = V2 con MINF = 12. Mismo calculo de 2024 frente a 2022-23 como referencia.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

D = os.path.join(ROOT, 'data')
df = pd.concat([pd.read_csv(os.path.join(D, f'session_{y}_V2.csv')).assign(Year=y) for y in range(2021, 2025)],
               ignore_index=True)
df = df[~df.Compound.isin(['WET', 'INTERMEDIATE'])].sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
ts = df.TrackStatus.astype(str).str.replace(r'\.0$', '', regex=True)
df['neutral'] = ts.str.contains('[4567]')
df['nlaps'] = df.groupby(['Year', 'GP']).LapNumber.transform('max')
g = df.groupby(['Year', 'GP', 'Driver'])
df['stint'] = g.PitIn.transform(lambda s: s.shift(fill_value=0).cumsum())     # la vuelta de entrada cierra su stint
df['neutral_prev'] = g.neutral.shift(fill_value=False)
df['drv_last'] = g.LapNumber.transform('max')
st = (df.groupby(['Year', 'GP', 'Driver', 'stint'])
        .agg(comp=('Compound_Detail', 'first'), ncomp=('Compound_Detail', 'nunique'), laps=('LapNumber', 'count'),
             first_lap=('LapNumber', 'first'), last_lap=('LapNumber', 'last'), tl_end=('TyreLife', 'last'),
             pit=('PitIn', 'last'), neutral_end=('neutral', 'last'), neutral_prev=('neutral_prev', 'last'),
             nlaps=('nlaps', 'first'), drv_last=('drv_last', 'first'))
        .reset_index().sort_values(['Year', 'GP', 'Driver', 'stint']))
# la parada que abrio el stint fue en verde? (cierre del stint anterior)
prev = st.groupby(['Year', 'GP', 'Driver'])
st['start_green'] = ~(prev.neutral_end.shift().fillna(True).astype(bool) | prev.neutral_prev.shift().fillna(True).astype(bool))
st['is_last'] = st.stint == prev.stint.transform('max')
st = st[(st.ncomp == 1) & st.comp.notna()]
st['finished'] = st.drv_last >= st.nlaps - 2


def sample(minf, years):
    mid = (st.pit == 1) & ~st.neutral_end & ~st.neutral_prev & (st.laps >= 8) & ~st.is_last
    # caso borde: ultimo stint que termina con PitIn (p. ej. parada en la ultima vuelta) cuenta como intermedio
    mid |= (st.pit == 1) & ~st.neutral_end & ~st.neutral_prev & (st.laps >= 8) & st.is_last & (st.last_lap < st.nlaps - 2)
    fin = st.is_last & st.finished & (st.pit == 0) & (st.stint > 0) & st.start_green & (st.laps >= minf)
    s = st[(mid | fin) & st.Year.isin(years)].copy()
    s['final'] = fin[mid | fin].astype(int).reindex(s.index)
    s['cell'] = s.GP + '|' + s.comp
    s = s[s.tl_end > 0]
    s['ltl'] = np.log(s.tl_end)
    return s


def factor(s, y, formula_extra='', by_comp=False):
    s = s.copy()
    both = s.groupby('cell').Year.agg(lambda t: (t == y).any() and (t != y).any())
    s = s[s.cell.map(both)]
    s['yy'] = (s.Year == y).astype(int)
    rhs = 'C(cell) + ' + ('yy:C(comp)' if by_comp else 'yy') + formula_extra
    f = smf.ols('ltl ~ ' + rhs, data=s).fit(cov_type='cluster', cov_kwds={'groups': s.GP})
    if by_comp:
        out = {}
        for k in f.params.index:
            if k.startswith('yy:'):
                lo, hi = f.conf_int().loc[k]
                out[k[-3:-1]] = (np.exp(-f.params[k]), np.exp(-hi), np.exp(-lo), int(((s.comp == k[-3:-1]) & (s.yy == 1)).sum()))
        return out
    lo, hi = f.conf_int().loc['yy']
    return dict(f=np.exp(-f.params['yy']), lo=np.exp(-hi), hi=np.exp(-lo), celdas=s.cell.nunique(),
                n_y=int(s.yy.sum()), n_ref=int((1 - s.yy).sum()), n_final_y=int((s.final * s.yy).sum()),
                n_final_ref=int((s.final * (1 - s.yy)).sum()))


V = {'V1 sin ultimos stints (M2)': (999, ''), 'V2 con ultimos stints': (8, ''),
     'V3 V2 + control ultimo stint': (8, ' + final'), 'V4 V2, ultimos stints >= 12': (12, '')}
rows = []
for name, (minf, extra) in V.items():
    for y in [2021, 2024]:
        s = sample(minf, [y, 2022, 2023])
        rows.append(dict(variante=name, anio=y, **factor(s, y, extra)))
R = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('Factor = (TyreLife al final del stint en 2022-23) / (en el anio), celdas GP x compuesto pareadas.')
print('Para 2021, el factor es lo que multiplica a TyreLife 2021; para 2024 es solo referencia (1 = igual que 2022-23).')
print(R.round(3).to_string(index=False))
R.to_csv(os.path.join(HERE, 'stint_factors_v2_20261015.csv'), index=False)

print('\nPor compuesto (2021, V2): factor, IC95, stints 2021')
for c, (f, lo, hi, n) in factor(sample(8, [2021, 2022, 2023]), 2021, by_comp=True).items():
    print(f'  {c}: {f:.3f} ({lo:.3f}-{hi:.3f}), n={n}')
s = sample(8, [2021, 2022, 2023])
print('\nStints en la muestra V2 por anio (intermedios / ultimos):')
print(s.groupby(['Year', 'final']).size().unstack().to_string())
print('\nTyreLife medio al final del stint, mismas celdas, por tipo:')
both = s.groupby('cell').Year.agg(lambda t: (t == 2021).any() and (t != 2021).any())
s = s[s.cell.map(both)]
print(s.assign(y=np.where(s.Year == 2021, '2021', '2022-23')).pivot_table(index='final', columns='y', values='tl_end',
                                                                          aggfunc='mean').round(1).to_string())
