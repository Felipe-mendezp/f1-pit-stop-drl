import re, glob, numpy as np, pandas as pd
rows=[]
for f in glob.glob('trained_models/retrained_v10eGQf_20261029/logs/*__PPO.log')+glob.glob('trained_models/retrained_v10eGQf_20261029/logs/*__RecurrentPPO.log'):
    _,s,gp,al=f.split('/')[-1][:-4].split('__')
    txt=open(f).read()
    ts=[int(x) for x in re.findall(r'total_timesteps\s+\|\s+(\d+)',txt)]
    en=[float(x) for x in re.findall(r'entropy_loss\s+\|\s+(-?[\d.e+-]+)',txt)]
    n=min(len(ts),len(en)); ts,en=np.array(ts[-n:]),-np.array(en[-n:])
    def at(t): return en[np.searchsorted(ts,t)] if ts[-1]>=t else np.nan
    t01=ts[np.argmax(en<0.01)] if (en<0.01).any() else np.nan
    rows.append(dict(gp=gp,algo=al,seed=s,H_250k=at(2.5e5),H_500k=at(5e5),H_1M=at(1e6),H_end=en[-1],t_H_lt_001=t01/1e6))
d=pd.DataFrame(rows); pd.set_option('display.width',200)
print(d.groupby(['algo','gp'])[['H_250k','H_500k','H_1M','H_end','t_H_lt_001']].median().round(3))
