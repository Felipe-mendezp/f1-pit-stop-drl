"""2026-10-28 — Compuesto por vuelta de carrera 2024: data/session_2024_V2.csv (FastF1) frente a la hoja 'OpenF1 Stints'
de data/f1_2024_tire_usage.xlsx. Motivo: en Belgica 2024 FastF1 marca MEDIUM/SOFT tandas que OpenF1 y Lapchive dan como
HARD (revision de GP_CONFIGS, docs/AUDITORIA_V10_20261028.md). Correr desde la raiz del repo.
Salida: mismatch_laps.csv (vueltas distintas), mismatch_by_race.csv
"""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
O = pd.read_excel('data/f1_2024_tire_usage.xlsx', 'OpenF1 Stints', header=3)
O = O[O['Sesión'] == 'Race'].dropna(subset=['Vuelta inicial', 'Vuelta final'])
s = pd.read_csv('data/session_2024_V2.csv')
# nombre completo -> codigo: por apellido
last = {r: r.split()[-1].upper()[:3] for r in O.Piloto.unique()}
fix = {'Nico Hulkenberg': 'HUL', 'Guanyu Zhou': 'ZHO', 'Kevin Magnussen': 'MAG', 'Andrea Kimi Antonelli': 'ANT'}
O['Driver'] = O.Piloto.map(lambda n: fix.get(n, last[n]))
rows = []
for r in O.itertuples(index=False):
    for lap in range(int(r[9]), int(r[10]) + 1):
        rows.append((r[1], r.Driver, lap, str(r[12]).upper()))
OL = pd.DataFrame(rows, columns=['GP', 'Driver', 'LapNumber', 'Compound_openf1']).drop_duplicates(['GP', 'Driver', 'LapNumber'], keep='last')
m = s.merge(OL, on=['GP', 'Driver', 'LapNumber'], how='left')
miss = m.Compound_openf1.isna()
print(f'vueltas FastF1: {len(m)}; sin par en OpenF1: {miss.sum()} (pilotos sin par: {sorted(m[miss].Driver.unique())[:15]})')
m = m[~miss]
bad = m[m.Compound != m.Compound_openf1]
bad.to_csv(os.path.join(HERE, 'mismatch_laps.csv'), index=False)
B = bad.groupby('GP').agg(laps=('LapNumber', 'size'), drivers=('Driver', lambda x: ','.join(sorted(set(x))))).reset_index()
B.to_csv(os.path.join(HERE, 'mismatch_by_race.csv'), index=False)
print(f'vueltas comparadas {len(m)}, distintas {len(bad)} ({len(bad)/len(m):.2%})')
print(bad.groupby(['Compound', 'Compound_openf1']).size().to_string())
pd.set_option('display.width', 250); print(B.to_string(index=False))
