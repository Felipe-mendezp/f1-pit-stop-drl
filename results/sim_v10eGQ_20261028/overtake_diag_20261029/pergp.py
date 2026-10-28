import os, numpy as np, pandas as pd
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
s=pd.read_pickle(T+'ot_gap.pkl'); full=pd.read_pickle(T+'ot_full.pkl')
R=pd.concat([pd.read_pickle(T+'otsim/ot_sim_phys.pkl'),pd.read_pickle(T+'otsim/ot_sim_phys2.pkl')])
real=s.groupby(['g','Year']).ot_pair.sum().groupby('g').mean()
f=full[(full.LapNumber>4)&(full.LapNumber<full.LastLap)&(full.TrackStatus.astype(str).isin(['1','1.0']))&full.interval_front_real.notna()&(full.Position>1)&(full.interval_front_real<60)]
f=f[f.Year.isin(s.Year.unique())]; f['g']=f.GP.str.replace(' Grand Prix','')
X=pd.DataFrame({'real':real})
for v in ['actual:E0','ab:E4']:
    X[v]=R[R['var']==v].groupby(['GP','seed']).ot.sum().groupby('GP').mean()
X['le1_real']=f.groupby('g').interval_front_real.apply(lambda x:(x<=1).mean())
for v in ['actual:E0','ab:E4']:
    q=R[(R['var']==v)&(R.gap<60)]; X['le1_'+v]=q.groupby('GP').gap.apply(lambda x:(x<=1).mean())
print(X.round(2).to_string())
