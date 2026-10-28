"""2026-10-28 — Validacion del inventario de juegos 2024 (supl. app. de juegos: "20% of the race stints started on a used set
(15% of the starts and 23% of the stints after a pit stop), and 236 of these 240 stints match a set of the inferred
inventory") con la definicion v10 de parada: en ../tyre_inventory_mixed_2024_20261021/ una tanda de carrera es
1 + PitOut acumulado, asi que un paso por el pit lane sin parada abre una tanda "con juego usado" falsa.
Aqui se quitan de race_stint_ages_2024_keep1.csv las tandas que empiezan tras un paso (PitLap = vuelta con in_pass,
../sim_v10_20261027/common_v10_20261027.py) y se re-corre validate() de available_regulation_20261021.py con el inventario
del simulador (available_regulation_2024_score_5.0_0.0.csv). No escribe en la carpeta original.
    python results/tyre_inventory_v10_20261028/validate_v10_20261028.py     (desde la raiz del repo)
Salida: race_stint_ages_2024_v10.csv, removed_pass_stints.csv, summary_v10.csv
"""
import os, sys, runpy
import pandas as pd
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'results', 'tyre_inventory_mixed_2024_20261021')
sys.path.insert(0, os.path.join(ROOT, 'results', 'sim_v10_20261027'))
from common_v10_20261027 import pass_flags

R = pd.read_csv(os.path.join(SRC, 'race_stint_ages_2024_keep1.csv'))
F = pass_flags((2024,))
P = F[F.in_pass][['GP', 'Driver', 'LapNumber']].rename(columns={'LapNumber': 'PitLap'}).astype({'PitLap': int})
m = R.merge(P.assign(pass_=True), on=['GP', 'Driver', 'PitLap'], how='left')
rm = m[m.pass_.fillna(False).astype(bool)].drop(columns='pass_')
Rv = m[~m.pass_.fillna(False).astype(bool)].drop(columns='pass_')
unmatched = P.merge(R, on=['GP', 'Driver', 'PitLap'], how='left', indicator=True).query('_merge == "left_only"')
print(f'pasos 2024: {len(P)}; tandas quitadas: {len(rm)}; pasos sin tanda en el archivo (lluvia, sustitutos o ultima vuelta): {len(unmatched)}')
rm.to_csv(os.path.join(HERE, 'removed_pass_stints.csv'), index=False)
Rv.to_csv(os.path.join(HERE, 'race_stint_ages_2024_v10.csv'), index=False)

cwd = os.getcwd(); os.chdir(SRC)
try:
    ns = runpy.run_path('available_regulation_20261021.py', run_name='_lib')
    G = pd.read_csv('available_regulation_2024_score_5.0_0.0.csv')
finally:
    os.chdir(cwd)
avail = {(g, d): list(zip(x.Compound, x.Age.astype(int))) for (g, d), x in G.groupby(['GP', 'Driver'])}
validate = ns['validate']
rows = []
for name, r in (('publicado', R), ('v10', Rv)):
    validate.__globals__['R'] = r
    v = validate(avail)
    used = r.Age > 0
    start = r.PitLap == 0
    rows.append(dict(version=name, stints=len(r), used=int(used.sum()), share_used=used.mean(),
                     share_used_starts=used[start].mean(), share_used_after_stop=used[~start].mean(),
                     used_stints_validate=v['used_stints'], used_unexplained=v['used_unexplained'],
                     matched=v['used_stints'] - v['used_unexplained'], new_short=v['new_short']))
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, 'summary_v10.csv'), index=False)
pd.set_option('display.width', 250); print(S.round(3).to_string(index=False))
