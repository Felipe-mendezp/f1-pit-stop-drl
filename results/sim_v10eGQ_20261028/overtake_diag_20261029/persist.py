import os, numpy as np, pandas as pd
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
full=pd.read_pickle(T+'ot_full.pkl')
k=full.set_index(['Year','GP','Driver','LapNumber'])
iv=k.interval_front_real.to_dict(); pos=k.Position.to_dict(); ts=k.TrackStatus.astype(str).to_dict()
pi=k.PitIn.to_dict(); po=k.PitOut.to_dict()
f=full[(full.LapNumber>4)&(full.LapNumber<full.LastLap-1)&(full.Position>1)&full.interval_front_real.notna()].copy()
f['iv1']=[iv.get((y,g,d,l+1),np.nan) for y,g,d,l in zip(f.Year,f.GP,f.Driver,f.LapNumber)]
f['p1']=[pos.get((y,g,d,l+1),np.nan) for y,g,d,l in zip(f.Year,f.GP,f.Driver,f.LapNumber)]
f['ts1']=[ts.get((y,g,d,l+1)) for y,g,d,l in zip(f.Year,f.GP,f.Driver,f.LapNumber)]
f['pit']=[bool(pi.get((y,g,d,l+1),0) or po.get((y,g,d,l+1),0)) for y,g,d,l in zip(f.Year,f.GP,f.Driver,f.LapNumber)]
g=f[(f.ts1.isin(['1','1.0']))&(f.TrackStatus.astype(str).isin(['1','1.0']))&(f.p1==f.Position)&~f.pit&f.iv1.notna()]
print('real: pares que mantienen posicion, sin parada, verde L y L+1: n',len(g))
for lo,hi in [(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,3)]:
    x=g[(g.interval_front_real>lo)&(g.interval_front_real<=hi)]
    print(f'  gap inicial ({lo},{hi}]: n {len(x):5d} | gap siguiente <=1s {(x.iv1<=1).mean():.0%} | mediana {x.iv1.median():.2f} | p10 {x.iv1.quantile(.1):.2f}')
print('simulador E0: un par que no adelanta termina en max(gap natural, 0.4+Exp(1)); P(0.4+Exp(1) <= 1) =', round(1-np.exp(-0.6),2))
gg=g.merge(s_ := pd.read_pickle(T+'ot_gap.pkl')[['Year','GP','Driver','LapNumber','delta_total']],on=['Year','GP','Driver','LapNumber'])
for lo,hi in [(0,1),(0,1.5)]:
    b=gg[(gg.interval_front_real>lo)&(gg.interval_front_real<=hi)&(gg.delta_total<0.4)]
    print(f'\nreal bloqueados (gap inicial ({lo},{hi}], delta medio < 0.4, no adelanta): n {len(b)}; gap siguiente percentiles',
          b.iv1.quantile([.05,.1,.25,.5,.75,.9]).round(2).to_dict(),'media',round(b.iv1.mean(),2))
    print('   ajuste 0.4+Exp(s): s = media-0.4 =',round(b.iv1.mean()-0.4,2),'| fraccion < 0.4:',f'{(b.iv1<0.4).mean():.0%}')
b=gg[(gg.interval_front_real>0)&(gg.interval_front_real<=1)&(gg.delta_total<0.4)]
np.save(T+'otsim/blocked_gaps.npy', b.iv1.values)
