import os, sys; ROOT=os.getcwd(); sys.path[:0]=[ROOT, os.path.join(ROOT,'src'), os.path.join(ROOT,'results','rival_model_eval_20261014')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, patsy
import rivals_common_20261014 as RC
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
    print('races',G); print(out.to_string())
    out.to_csv(os.environ['OUT'])
main()
