import os, numpy as np, pandas as pd
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
s=pd.read_pickle(T+'ot_gap.pkl'); full=pd.read_pickle(T+'ot_full.pkl'); R=pd.concat([pd.read_pickle(T+'otsim/ot_sim_phys.pkl'),pd.read_pickle(T+'otsim/ot_sim_phys2.pkl'),pd.read_pickle(T+'otsim/ot_sim_phys3.pkl'),pd.read_pickle(T+'otsim/ot_sim_final.pkl')])
f=full[(full.LapNumber>4)&(full.LapNumber<full.LastLap)&(full.TrackStatus.astype(str).isin(['1','1.0']))&full.interval_front_real.notna()&(full.Position>1)&(full.interval_front_real<60)]
band=lambda g: pd.cut(g,[-np.inf,1,2,np.inf],labels=['<=1s','1-2s','>2s'])
s['band']=band(s.interval_front_real); R['band']=band(R.gap)
rows=[dict(var='real',le1=(f.interval_front_real<=1).mean(),p10=f.interval_front_real.quantile(.1),p50=f.interval_front_real.median(),
           **s.groupby(['g','Year','band'],observed=False).ot_pair.sum().groupby('band',observed=False).sum().div(s.groupby(['g','Year']).ngroups/8).to_dict(),
           rate_le1=s[s.band=='<=1s'].ot_pair.mean(),rate_12=s[s.band=='1-2s'].ot_pair.mean())]
for v,q in R.groupby('var',sort=False):
    qq=q[q.gap<60]; c=q.groupby(['band'],observed=False).ot.sum()/q.groupby(['GP','seed']).ngroups*8
    rows.append(dict(var=v,le1=(qq.gap<=1).mean(),p10=qq.gap.quantile(.1),p50=qq.gap.median(),**c.to_dict(),
                     rate_le1=q[q.band=='<=1s'].ot.mean(),rate_12=q[q.band=='1-2s'].ot.mean()))
X=pd.DataFrame(rows); X['total']=X[['<=1s','1-2s','>2s']].sum(1)
print('adelantamientos: suma de los 8 GP de la media por carrera; gaps: pares adyacentes al inicio de vuelta verde, vueltas 5+')
print(X.round(3).to_string(index=False))
