"""2026-10-14: Fig. 6 v5h con abandonos reales de 2024 (--real-dnf) frente a la base sin abandonos
(results/world2024_v5h_20261012/fig6_2024; misma semilla 42 y 500 carreras). Para los 4 GP con conteo de
paradas (world2024_v5h_pits_20261014) compara tambien el % de carreras con parada en las ultimas 6 vueltas."""
import os, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
NEW = os.path.join(HERE, 'fig6_2024_realdnf')
BASE = os.path.join(R, 'world2024_v5h_20261012', 'fig6_2024')
PITS = os.path.join(R, 'world2024_v5h_pits_20261014', 'fig6_2024')
N_LAPS = {'Bahrain_ALO': 57, 'Saudi_Arabian_NOR': 50, 'Miami_OCO': 57, 'Hungarian_SAI': 70, 'Belgian_NOR': 44,
          'Dutch_RUS': 72, 'Singapore_HUL': 62, 'United_States_PIA': 56}


def late(df, n):
    return df.pit_laps.fillna('').astype(str).apply(lambda s: any(int(float(x)) > n - 6 for x in s.split())).mean() * 100


rows = []
for f in sorted(glob.glob(os.path.join(NEW, '*.csv'))):
    gp = os.path.basename(f)[:-4]
    new, base = pd.read_csv(f), pd.read_csv(os.path.join(BASE, gp + '.csv'))
    pits = pd.read_csv(os.path.join(PITS, gp + '.csv')) if os.path.exists(os.path.join(PITS, gp + '.csv')) else None
    for algo in new.algo.unique():
        a, b = new[new.algo == algo], base[base.algo == algo]
        r = dict(gp=gp, algo=algo, pos_base=b.final_position.mean(), pos_realdnf=a.final_position.mean(),
                 pits_base=b.pit_stops.mean(), pits_realdnf=a.pit_stops.mean(),
                 late_realdnf=late(a, N_LAPS[gp]))
        if pits is not None:
            r['late_base'] = late(pits[pits.algo == algo], N_LAPS[gp])
        rows.append(r)
df = pd.DataFrame(rows)
df['d_pos'] = df.pos_realdnf - df.pos_base
df.to_csv(os.path.join(HERE, 'compare_20261014.csv'), index=False)
pd.set_option('display.width', 200)
print(df.round(2).to_string(index=False))
print('\nCambio medio de posicion por metodo (realdnf - base):')
print(df.groupby('algo').d_pos.mean().round(2).to_string())
