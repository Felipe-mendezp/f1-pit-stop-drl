import pandas as pd, glob, numpy as np, os
T='results/miami_lap6_diag_20261029/data/'
d=pd.concat([pd.read_csv(f) for f in glob.glob(T+'part*.csv')])
a=pd.read_csv(T+'agents.csv'); a=a[~a.strat.str.contains('-')]
d[['s','L','c']]=d.strat.str.split('-',expand=True); d.L=d.L.astype(int)
g=d.groupby('strat').agg(L=('L','first'),s=('s','first'),c=('c','first'),R=('reward','mean'),pos=('pos','mean'),dsq=('ok',lambda x:1-x.mean()),realL=('pit_lap','mean')).reset_index()
pd.set_option('display.width',200)
print('mejor por vuelta (recompensa):'); print(g.loc[g.groupby('L').R.idxmax()][['L','strat','R','pos','realL']].round(2).to_string(index=False))
print('top 8 por recompensa:'); print(g.sort_values('R',ascending=False).head(8).round(2).to_string(index=False))
print('top 8 por posicion:'); print(g.sort_values('pos').head(8).round(2).to_string(index=False))
print(a.groupby('strat').agg(R=('reward','mean'),pos=('pos','mean')).round(2))
for s in a.strat.unique(): print(s, a[a.strat==s].pits.str.split().str[:3].str.join(' ').value_counts().head(3).to_dict())
# pareado: mejor fija vs 4-6-3
b=g.sort_values('R',ascending=False).strat.iloc[0]
x=d[d.strat==b].set_index('sim'); y=d[d.strat=='4-6-3'].set_index('sim')
dd=x.reward-y.reward; dp=x.pos-y.pos
print(f'{b} - 4-6-3: dR {dd.mean():.1f} ± {1.96*dd.std()/np.sqrt(len(dd)):.1f}, dpos {dp.mean():.2f} ± {1.96*dp.std()/np.sqrt(len(dp)):.2f}')
g.to_csv(T+'summary.csv',index=False)
