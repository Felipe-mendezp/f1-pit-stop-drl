import numpy as np, glob, os, re, pandas as pd
T='trained_models/retrained_v10eGQf_20261029'
rows=[]
for f in sorted(glob.glob(f'{T}/seed*/*/*/evaluations.npz')):
    s,gp,al=f.split('/')[-4:-1]
    d=np.load(f); r=d['results'].mean(1); t=d['timesteps']
    ib=r.argmax()
    rows.append(dict(gp=gp,algo=al,seed=int(s[4:]),best=r.max(),t_best=t[ib]/1e6,final=r[-1],r_1M=r[np.searchsorted(t,1e6)] if t[-1]>=1e6 else np.nan,n=len(t)))
df=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(df.groupby(['gp','algo'])[['best','t_best','final','r_1M']].mean().round(1).unstack('algo')['best'])
print(df.groupby(['gp','algo'])['t_best'].mean().round(2).unstack())
print(df[df.gp=='Miami_OCO'].sort_values(['algo','seed']).round(1).to_string(index=False))
