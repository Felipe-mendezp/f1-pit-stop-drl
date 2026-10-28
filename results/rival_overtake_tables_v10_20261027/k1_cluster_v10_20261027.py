"""2026-10-27 — K1 v10 (paradas reales, results/sim_v10_20261027/rivals_v10_20261027.py) con e.e. agrupados por carrera
y pseudo R2 de McFadden frente a probabilidades iguales. Copia de ../rival_overtake_tables_v7_20261020/k1_cluster_20261020.py
con las vueltas de common_v10_20261027.rival_laps (PitIn = parada real).
    F1_TYRELIFE_2021=v10 OUT=results/rival_overtake_tables_v10_20261027/k1_cluster.csv python results/rival_overtake_tables_v10_20261027/k1_cluster_v10_20261027.py
"""
import os, sys; ROOT=os.getcwd(); sys.path[:0]=[ROOT, os.path.join(ROOT,'src'), os.path.join(ROOT,'results','rival_model_eval_20261014'), os.path.join(ROOT,'results','sim_v10_20261027')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, patsy
import rivals_common_20261014 as RC
import common_v10_20261027 as CV
from models import rival_models as RM
_orig = RM.load_lap_data
def _load():
    RM.load_lap_data = _orig
    try:
        return CV.rival_laps(RM).drop(columns=['PitIn_raw'])
    finally:
        RM.load_lap_data = _load
RM.load_lap_data = _load
assert os.environ.get('F1_TYRELIFE_2021') == 'v10'
def main():
    train,_=RC.load(); L=RC.long_df(train); m=RC.K1(L); f=m.fit
    X=patsy.build_design_matrices([m.info], L, return_type='dataframe')[0][m.cols].values
    b=f.params.values if hasattr(f.params,'values') else np.asarray(f.params)
    v=X@b; g=L.decision_id.values
    df=pd.DataFrame(dict(g=g,v=v)); e=np.exp(v-df.groupby('g').v.transform('max').values)
    p=e/pd.Series(e).groupby(g).transform('sum').values
    y=L.chosen.values
    # per-decision score and Hessian
    S=pd.DataFrame(X*(y-p)[:,None]).groupby(g).sum()
    H=np.zeros((X.shape[1],)*2)
    for _,idx in pd.Series(np.arange(len(g))).groupby(g):
        xi=X[idx.values]; pi=p[idx.values]; xb=pi@xi
        H+= (xi*pi[:,None]).T@xi - np.outer(xb,xb)
    Hi=np.linalg.inv(H)
    race=L.groupby('decision_id').apply(lambda d: f"{d.Year.iloc[0]}_{d.GP.iloc[0]}").reindex(S.index).values
    Sc=S.groupby(race).sum().values; G=Sc.shape[0]
    B=Sc.T@Sc*G/(G-1)
    se_cl=np.sqrt(np.diag(Hi@B@Hi)); se=np.sqrt(np.diag(Hi))
    from scipy.stats import norm
    out=pd.DataFrame(dict(b=b,se=se,se_sm=np.asarray(f.bse),se_cl=se_cl,p_cl=2*norm.sf(np.abs(b/se_cl))),index=m.cols)
    J=L.groupby('decision_id').size(); ll=float((np.log(p)*y).sum()); ll0=float(-np.log(J).sum())
    print('races',G,'decisiones',len(J),'pseudo R2',round(1-ll/ll0,3)); print(out.to_string())
    open(os.environ['OUT'].replace('.csv','_meta.txt'),'w').write(f'races {G}\ndecisions {len(J)}\npseudo_r2 {1-ll/ll0:.6f}\n')
    out.to_csv(os.environ['OUT'])
main()
