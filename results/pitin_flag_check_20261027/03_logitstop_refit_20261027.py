import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path[:0]=['.', 'src']
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from models import rival_models as RM
L=RM.load_lap_data().sort_values(['Year','GP','Driver','LapNumber'])
k=['Year','GP','Driver']; n=L.groupby(k).shift(-1)
L['stop']=(((n.TyreLife<L.TyreLife)|(n.Compound_Detail!=L.Compound_Detail))&(n.LapNumber==L.LapNumber+1)).astype(int)
# SC event index per race-lap (majority of drivers with '4' in TrackStatus)
ts=L.TrackStatus.astype(str)
lap=L.assign(sc=ts.str.contains('4')).groupby(['Year','GP','LapNumber']).sc.mean().gt(0.5).reset_index()
idx=[];prev=None;c=0
for r in lap.itertuples():
    if (r.Year,r.GP)!=prev: prev=(r.Year,r.GP); c=0
    c=c+1 if r.sc else 0; idx.append(c)
lap['sc_idx']=idx
L=L.merge(lap[['Year','GP','LapNumber','sc_idx']],on=['Year','GP','LapNumber'])
X=RM.preprocess_laps(L)
f="PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP"
res={}
for name,col in [('original',X.PitIn),('PitIn&stop',X.PitIn*X.stop),('stop',X.stop)]:
    Y=X.assign(PitIn=col.values)
    r=smf.logit(f,Y).fit(disp=False); res[name]=r
    first=Y[(Y.SC==1)&(Y.sc_idx==1)&(Y.LapNumber>5)]
    print(f"{name:11s} n_pits={int(Y.PitIn.sum())} SC={r.params['SC']:.3f}({r.bse['SC']:.3f}) VSC={r.params['VSC']:.3f}({r.bse['VSC']:.3f}) "
          f"| 1st SC lap: mean pred prob {r.predict(first).mean():.3f} vs real stop rate {first.stop.mean():.3f} (n={len(first)})")
