"""2026-10-14 — filas LaTeX de tab:regression_cleanlap (US) desde table_us.csv. Salida: table_us_rows.tex"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(HERE, 'table_us.csv'), header=[0, 1], index_col=0)
NAMES = dict(ALB='Albon', ALO='Alonso', BOT='Bottas', GAS='Gasly', GIO='Giovinazzi', HAM='Hamilton', HUL='Hülkenberg',
             LAT='Latifi', LEC='Leclerc', MAG='Magnussen', MAZ='Mazepin', MSC='Schumacher', NOR='Norris', OCO='Ocon',
             PER='Pérez', PIA='Piastri', RAI='Räikkönen', RIC='Ricciardo', RUS='Russell', SAI='Sainz', SAR='Sargeant',
             STR='Stroll', TSU='Tsunoda', VET='Vettel', ZHO='Zhou')
ORDER = [('Intercept', 'Intercept'), ('DRS', 'DRS'), ('FirstLap_pos', r'FirstLap\_pos'),
         ('Interval_behind', r'Interval\_behind'), ('Interval_front', r'Interval\_front'), ('LapNumber', 'LapNumber'),
         ('LapNumber_2', r'LapNumber$^2$'), ('PitIn', 'PitIn'), ('PitOut', 'PitOut'), ('C3', 'C3 (ref. C2)'),
         ('C4', 'C4 (ref. C2)'), ('Year 2022', 'Year 2022'), ('Year 2023', 'Year 2023')] + \
        [(f'TireLife x {c}', f'TireLife $\\cdot$ {c}') for c in ('C2', 'C3', 'C4')] + \
        sorted([(f'drv:{k}', v) for k, v in NAMES.items()], key=lambda x: x[1].replace('Hü', 'Hu').replace('Pé', 'Pe').replace('Rä', 'Ra'))
assert len(ORDER) == len(T), (len(ORDER), len(T))
f = lambda r, k: f"{r[(k, 'coef')]:.4f} & {r[(k, 'sig')] if isinstance(r[(k, 'sig')], str) else ''} & ({r[(k, 'se')]:.3f})"
lines = [f"{lab} & {f(T.loc[i], 'with')} & {f(T.loc[i], 'without')} \\\\" for i, lab in ORDER]
open(os.path.join(HERE, 'table_us_rows.tex'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
