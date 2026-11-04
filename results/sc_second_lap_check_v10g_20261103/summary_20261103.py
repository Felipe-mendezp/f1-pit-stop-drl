"""2026-11-03 — Resumen del chequeo en v10g: puntos y posicion por brazo y diferencia pareada frente a medio (+-2 SE);
control de los 4 brazos contra el barrido. Escribe summary_20261103.csv."""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(H))
d = pd.read_csv(f'{H}/fig8/saudi_arabian_nor_sc2_lap7.csv')
p = pd.read_csv(f'{ROOT}/results/decision_support_v10gw24belief_20261103/PPO/saudi_arabian_nor_real_lap7.csv')
m = d[d.action < 10].merge(p, on=['action', 'rollout'], suffixes=('', '_p'))
print('control 4 brazos:', len(m), 'filas; posicion igual', (m.final_position == m.final_position_p).mean(), '; dt max', (m.total_time - m.total_time_p).abs().max())
P = d.pivot(index='rollout', columns='action', values='points'); rows = []
for a in P:
    z = P[a] - P[2]; g = d[d.action == a]
    rows.append(dict(action=a, label=g.action_label.iloc[0], pts=g.points.mean(), pos=g.final_position.mean(), top10=(g.final_position <= 10).mean(),
                     d_vs_medium=z.mean(), se2=2 * z.std(ddof=1) / np.sqrt(len(z))))
S = pd.DataFrame(rows); S.to_csv(f'{H}/summary_20261103.csv', index=False); print(S.round(2).to_string(index=False))
for a, b in [(12, 0), (11, 0), (13, 0)]:
    if a in P: z = P[a] - P[b]; print(a, 'vs', b, round(z.mean(), 2), '+-', round(2 * z.std(ddof=1) / np.sqrt(len(z)), 2))
