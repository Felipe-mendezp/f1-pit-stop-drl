"""2026-10-30 — Cobertura de la validacion con la estrategia real (fig7) por variante del adelantamiento. Reusa
actual() y pit() de results/real_strategy_check_v10eGQf_20261029/compare_20261029.py. Variante 'vigente' = la carpeta
fig7 de esa validacion (v10f con el envoltorio, = l000).
    python results/overtake_noise_frac_20261030/compare_lambda_20261030.py l050 l100 ...
"""
import os, sys, importlib.util, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
REF = os.path.join(ROOT, 'results', 'real_strategy_check_v10eGQf_20261029')
spec = importlib.util.spec_from_file_location('cmp', os.path.join(REF, 'compare_20261029.py'))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
s = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'), usecols=['GP', 'Driver', 'LapNumber', 'Position'])
summ, per_gp = [], []
for v in ['vigente'] + sys.argv[1:]:
    d0 = REF if v == 'vigente' else os.path.join(HERE, v)
    rows = []
    for key, gp in C.GPS.items():
        d = pd.read_csv(os.path.join(d0, os.environ.get('SCEN', 'fig7'), f'{key}.csv')); act = C.actual(gp, s)
        for drv, x in d.groupby('driver'):
            pos = x.position.to_numpy(); a, fin = act.loc[drv, 'actual'], act.loc[drv, 'finished']
            q05, q50, q95 = np.quantile(pos, [0.05, 0.5, 0.95], method='inverted_cdf')
            rows.append(dict(gp=key, driver=drv, agent=bool(x.agent.iloc[0]), actual=a, finished=fin,
                             sim_dnf=x.is_dnf.mean(), mean=pos.mean(), sd=pos.std(), q05=q05, q50=q50, q95=q95,
                             inside90=q05 <= a <= q95, pit=C.pit(pos, a)))
    R = pd.DataFrame(rows); A = R[R.agent]; G = R[R.finished & (R.sim_dnf <= 0.05)]
    summ.append(dict(var=v, n=len(G), cover90=G.inside90.mean(), width90=(G.q95 - G.q05).mean(), sd=G.sd.mean(),
                     mae_median=(G.q50 - G.actual).abs().mean(),
                     spearman=np.mean([spearmanr(h['mean'], h.actual).statistic for _, h in G.groupby('gp')]),
                     pit_quint=' '.join(map(str, np.histogram(G.pit, bins=5, range=(0, 1))[0])),
                     agent_in=int(A.inside90.sum()), agent_optim=(A.actual - A['mean']).mean()))
    for key, h in G.groupby('gp', sort=False):
        a = A[A.gp == key].iloc[0]
        per_gp.append(dict(var=v, gp=key, cover90=h.inside90.mean(), width90=(h.q95 - h.q05).mean(),
                           agent=f"{a.driver} real {a.actual} sim {a['mean']:.2f}±{a.sd:.2f} [{a.q05:.0f},{a.q95:.0f}]"))
    R.to_csv(os.path.join(HERE, f"compare_all_{os.environ.get('SCEN', 'fig7')}_{v}.csv"), index=False)
S = pd.DataFrame(summ); P = pd.DataFrame(per_gp)
print(S.round(3).to_string(index=False)); print()
print(P.pivot(index='gp', columns='var', values='cover90').round(2).to_string()); print()
print(P.pivot(index='gp', columns='var', values='width90').round(1).to_string()); print()
print(P.pivot(index='gp', columns='var', values='agent').to_string())
