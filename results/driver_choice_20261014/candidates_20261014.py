"""Candidatos a piloto del agente multi-piloto por GP (2024) — 2026-10-14.

Para cada GP del paper y cada piloto de la carrera 2024 (data/session_2024_V2.csv):
  start      = posicion de partida (config.json del simulador v5h8, initial_positions_2024)
  finish     = Position en la ultima vuelta registrada; laps = vueltas completadas vs n_laps
  coef       = coeficiente de piloto del simulador v5h8 (s/vuelta vs VER; NaN = sin coeficiente)
  anom       = vueltas en bandera verde (TrackStatus == 1), sin PitIn/PitOut y despues de la 1, con tiempo
               > mediana de esas vueltas del piloto + 3 s (no explicadas por parada, SC/VSC/amarilla o largada)
  max_excess = mayor exceso sobre esa mediana entre todas las vueltas limpias (s)
Escribe candidates_20261014.csv.
"""
import os, sys, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
SIM = os.path.join(ROOT, 'trained_models', 'simulation_v5h8_20261012')
GPS = ['Bahrain', 'Saudi Arabian', 'Miami', 'Hungarian', 'Belgian', 'Dutch', 'Singapore', 'United States']
d = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))
rows = []
for g in GPS:
    gp = g + ' Grand Prix'
    cfg = json.load(open(os.path.join(SIM, gp, 'config.json')))
    pred = pickle.load(open(os.path.join(SIM, gp, 'laptime_clear.pkl'), 'rb'))
    coef = {n.split('[T.')[1][:-1]: float(c) for n, c in zip(pred.feature_names_, pred.coef_) if 'C(Driver' in n}
    coef[pred.reference_categories_.get('Driver', 'VER')] = 0.0
    x = d[d.GP == gp]
    for drv, pos0 in cfg['initial_positions_2024'].items():
        k = drv.replace('Driver_', '')
        l = x[x.Driver == k].sort_values('LapNumber')
        fin = l.dropna(subset=['Position']).Position.iloc[-1] if len(l) else np.nan
        clean = l[(l.TrackStatus.astype(str) == '1') & (l.PitIn == 0) & (l.PitOut == 0) & (l.LapNumber > 1)].dropna(subset=['LapTime'])
        med = clean.LapTime.median()
        exc = clean.LapTime - med
        rows.append(dict(GP=g, driver=k, start=pos0, finish=fin, laps=int(l.LapNumber.max()) if len(l) else 0,
                         n_laps=cfg['n_laps'], coef=coef.get(k, np.nan), n_clean=len(clean),
                         anom=int((exc > 3).sum()), max_excess=round(float(exc.max()), 2) if len(clean) else np.nan,
                         stops=int(l.PitIn.sum() - (l.LapNumber.max() == l[l.PitIn == 1].LapNumber.max() if (l.PitIn == 1).any() else 0))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, 'candidates_20261014.csv'), index=False)
pd.set_option('display.width', 200)
for g in GPS:
    t = df[df.GP == g].sort_values('start')
    print(f'\n== {g}'); print(t.drop(columns='GP').to_string(index=False))
