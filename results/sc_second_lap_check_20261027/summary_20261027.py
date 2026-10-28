"""2026-10-27 — Resumen del chequeo (b): por caso, media de puntos y posicion por brazo y diferencia pareada de puntos
frente al brazo recomendado en el paper (+-2 SE). Escribe summary_20261027.csv.
    ~/.venvs/f1-drl-v2/bin/python results/sc_second_lap_check_20261027/summary_20261027.py"""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(H))
CASES = [('fig8', 'saudi_arabian_nor_sc2_lap7', 'results/decision_support_v9r2w24belief_20261023/PPO/saudi_arabian_nor_real_lap7.csv'),
         ('fig11', 'saudi_arabian_nor_sc2_lap7', 'results/decision_support_v9r2smrealhist_20261026/PPO/saudi_arabian_nor_real_lap7.csv'),
         ('fig11', 'miami_oco_sc2_lap28', 'results/decision_support_v9r2smrealhist_20261026/PPO/miami_oco_real_lap28.csv'),
         ('fig11', 'united_states_mag_sc2_lap3', 'results/decision_support_v9r2smrealhist_20261026/PPO/united_states_mag_real_lap3.csv')]
rows = []
for fig, tag, pub in CASES:
    d = pd.read_csv(os.path.join(H, fig, tag + '.csv'))
    p = pd.read_csv(os.path.join(ROOT, pub)) if os.path.exists(os.path.join(ROOT, pub)) else None
    piv = d.pivot(index='rollout', columns='action', values='points')
    ref = piv.drop(columns=[c for c in piv if c >= 10]).mean().idxmax()
    for a in piv:
        diff = piv[a] - piv[ref]
        g = d[d.action == a]
        pp = p[p.action == a].points.mean() if (p is not None and a < 10) else np.nan
        rows.append(dict(fig=fig, case=tag, action=a, label=g.action_label.iloc[0], pts=g.points.mean(),
                         pos=g.final_position.mean(), pts_paper_rule=pp, ref=ref,
                         d_vs_ref=diff.mean(), se2=2 * diff.std(ddof=1) / np.sqrt(len(diff))))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(H, 'summary_20261027.csv'), index=False)
print(S.round(2).to_string(index=False))
