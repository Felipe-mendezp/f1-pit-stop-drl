import pandas as pd, numpy as np
out=[]
for y in (2021,2022,2023,2024):
    s=pd.read_csv(f'data/session_{y}_V2.csv').sort_values(['GP','Driver','LapNumber'])
    s['ts']=s.TrackStatus.astype(str)
    s['st']=np.select([s.ts.str.contains('5'),s.ts.str.contains('4'),s.ts.str.contains('6|7')],['RED','SC','VSC'],'G')
    n=s.groupby(['GP','Driver']).shift(-1)
    s['stop']=((n.TyreLife<s.TyreLife)|(n.Compound_Detail!=s.Compound_Detail))&(n.LapNumber==s.LapNumber+1)
    p=s.groupby(['GP','Driver']).shift(1)
    s['stop_prev']=((s.TyreLife<p.TyreLife)|(s.Compound_Detail!=p.Compound_Detail))&(p.LapNumber==s.LapNumber-1)
    s['Year']=y; out.append(s)
d=pd.concat(out)
d=d[~d.Compound.isin(['WET','INTERMEDIATE'])&(d.LapNumber>1)]
print('PitIn=1: share with a real stint change at that lap, by state')
print(d[d.PitIn==1].groupby('st').stop.agg(['size','mean']).round(3))
print('PitOut=1: share with stint change entering that lap')
print(d[d.PitOut==1].groupby('st').stop_prev.agg(['size','mean']).round(3))
print('real stops: share with PitIn=1, by state'); print(d[d.stop].groupby('st').PitIn.agg(['size','mean']).round(3))
bad=d[(d.PitIn==1)&~d.stop]
print('bad PitIn by year/state'); print(bad.groupby(['Year','st']).size().unstack(fill_value=0))
print(bad.groupby(['Year','GP']).size().sort_values(ascending=False).head(15))
print('\n=== 2021-23, logit-like sample (no red, dry, lap>1) ===')
e=d[(d.Year<=2023)&(d.st!='RED')]
print(e.groupby('st').agg(n=('PitIn','size'),pitin_rate=('PitIn','mean'),true_stop_rate=('stop','mean')).round(4))
