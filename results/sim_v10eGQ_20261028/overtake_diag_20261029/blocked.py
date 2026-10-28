import os, numpy as np, pandas as pd
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
s=pd.read_pickle(T+'ot_gap.pkl'); full=pd.read_pickle(T+'ot_full.pkl')
k=full.set_index(['Year','GP','Driver','LapNumber']); iv=k.interval_front_real.to_dict()
s['iv_next']=[iv.get((y,g,d,l+1),np.nan) for y,g,d,l in zip(s.Year,s.GP,s.Driver,s.LapNumber)]
# bloqueado: no adelanta, mismo rival delante, el de atras era mas rapido en la prediccion (delta medio < gap inicial)
b=s[(s.ot_pair==0)&(s.delta_t_lap<0)&(s.interval_front_real<=3)&s.iv_next.notna()]
print('real, pares bloqueados (no adelanta, era mas rapido, gap inicial<=3s): n',len(b))
print('  gap al final de la vuelta: percentiles',b.iv_next.quantile([.1,.25,.5,.75,.9]).round(2).to_dict(),'media',round(b.iv_next.mean(),2))
e=0.4+np.random.default_rng(0).exponential(1,100000)
print('  simulador 0.4+Exp(1): percentiles',{q:round(np.quantile(e,q),2) for q in [.1,.25,.5,.75,.9]},'media',round(e.mean(),2))
f=full[(full.LapNumber>4)&(full.LapNumber<full.LastLap)&(full.TrackStatus.astype(str).isin(['1','1.0']))&full.interval_front_real.notna()&(full.Position>1)]
f=f[f.interval_front_real<60]
R=pd.read_pickle(T+'otsim/ot_sim.pkl'); q=R[(R['var']=='actual')&(R.gap<60)]
print('gap adyacente al inicio de vuelta, sin filtro de delta (real verde, vueltas 5+):', f.interval_front_real.quantile([.1,.25,.5,.75]).round(2).to_dict(), '| <=1s:', f'{(f.interval_front_real<=1).mean():.1%}')
print('simulador actual:', q.gap.quantile([.1,.25,.5,.75]).round(2).to_dict(), '| <=1s:', f'{(q.gap<=1).mean():.1%}')
