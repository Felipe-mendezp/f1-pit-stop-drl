import os, numpy as np, pandas as pd, statsmodels.api as sm
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/'); s=pd.read_pickle(T+'otsim/sampleC.pkl')
F={'lin':lambda d:np.column_stack([np.ones_like(d),d]),'asinh':lambda d:np.column_stack([np.ones_like(d),np.arcsinh(d)])}
for k,f in F.items():
    s[k]=np.nan
    for g,x in s.groupby('g'):
        s.loc[x.index,k]=sm.Logit(x.ot_pair.values,f(x.delta_total.values)).fit(disp=False).predict(f(x.delta_total.values))
s['bin']=pd.cut(s.delta_total,[-30,-10,-5,-2,-1,-0.5,0,0.5,1,2,5])
print(s.groupby('bin',observed=False).agg(n=('ot_pair','size'),ot=('ot_pair','sum'),real=('ot_pair','mean'),lin=('lin','mean'),asinh=('asinh','mean')).round(3).T.to_string())
