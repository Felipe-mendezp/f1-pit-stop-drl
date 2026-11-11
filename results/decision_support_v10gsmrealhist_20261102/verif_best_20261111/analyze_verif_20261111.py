"""2026-11-11 — Verificacion de la columna 'mejor de las cuatro': compara, en las 28 vueltas de desacuerdo, la diferencia
de puntos con la decision real en el barrido original (semillas 1000..1499) y en la repeticion independiente
(semillas 100000..100499, mismo estado). La accion 'mejor' se elige en el barrido original y se mide en la repeticion.
    python results/decision_support_v10gsmrealhist_20261102/verif_best_20261111/analyze_verif_20261111.py
"""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H)
L = pd.read_csv(os.path.join(R, 'compare_laps_20261103.csv'))
L = L[(L.algo == 'PPO') & (L.scen == '(ii)') & (L.real != L.rec)]


def diff(df, a, b):
    p = df.pivot(index='rollout', columns='action_label', values='points')
    d = p[a] - p[b]
    return d.mean(), 2 * d.std(ddof=1) / np.sqrt(len(d)), p.mean().idxmax()


rows = []
for r in L.itertuples():
    f = f'{r.gp}_real_lap{r.lap}.csv'
    o = pd.read_csv(os.path.join(R, 'PPO', f)); n = pd.read_csv(os.path.join(H, 'PPO', f))
    dro, sro, _ = diff(o, r.rec, r.real); dbo, sbo, bo = diff(o, r.best_pts, r.real)
    drn, srn, _ = diff(n, r.rec, r.real); dbn, sbn, bn = diff(n, r.best_pts, r.real)
    rows.append(dict(gp=r.gp, lap=r.lap, real=r.real, rec=r.rec, best=r.best_pts, best_label_new=bn,
                     drl_orig=dro, drl_orig_2se=sro, drl_new=drn, drl_new_2se=srn,
                     best_orig=dbo, best_orig_2se=sbo, best_new=dbn, best_new_2se=sbn,
                     check_drl=abs(dro + r.dpts_real_vs_rec) < 1e-6, check_best=abs(dbo - r.gain_best_vs_real) < 1e-6,
                     check_best_label=bo == r.best_pts))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(H, 'verif_best_20261111.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(T[['gp', 'lap', 'real', 'rec', 'best', 'best_label_new', 'drl_orig', 'drl_new', 'drl_new_2se', 'best_orig', 'best_new',
         'best_new_2se']].round(2).to_string(index=False))
print('chequeos con el CSV original:', T.check_drl.all(), T.check_best.all(), T.check_best_label.all())
for c in ['drl_orig', 'drl_new', 'best_orig', 'best_new']:
    print(c, round(T[c].sum(), 2))
cat = np.where(T.best == T.rec, 'DRL', np.where(T.best == T.real, 'piloto', 'otra'))
print(pd.Series(cat).value_counts().to_dict(), '| mejor estable en la repeticion:', int((T.best == T.best_label_new).sum()), 'de', len(T))
