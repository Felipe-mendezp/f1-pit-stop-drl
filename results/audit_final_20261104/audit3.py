import pandas as pd, numpy as np, glob, os, json, re
V='results/overtake_noise_frac_20261030/l000_r28_k/fig7/'; R='results/world2024_v10g_20261031/fig7_2024/'
real=pd.read_csv('data/session_2024_V2.csv',low_memory=False); print(list(real.columns)[:40])
rows=[];cov=[];sp=[];opt=[]
from scipy.stats import spearmanr
for f in sorted(glob.glob(V+'*.csv')):
    g=os.path.basename(f)[:-4]; d=pd.read_csv(f); ag=d[d.agent==True]
    m=pd.read_csv(R+g+'.csv'); mm=m.groupby('algo').final_position.mean()
    rows.append(dict(g=g,replay=ag.position.mean(),sd=ag.position.std(),q05=ag.position.quantile(.05),q95=ag.position.quantile(.95),n=ag.sim.nunique(),best=mm.min(),bestalgo=mm.idxmin()))
print(pd.DataFrame(rows).round(2).to_string())
T=pd.DataFrame(rows); print('replay avg',T.replay.mean().round(3))
for a in ['PPO','A2C','DQN','TRPO','RecurrentPPO','QMIP']:
    v=np.mean([pd.read_csv(R+r['g']+'.csv').query('algo==@a').final_position.mean() for r in rows]); print(a,'improve',round(T.replay.mean()-v,3))
