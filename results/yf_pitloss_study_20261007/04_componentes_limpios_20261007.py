from yf_common_20261007 import *
d=load(); ws=d.groupby(['Year','GP'])['wet'].mean(); d=d.join(ws.rename('ws'),on=['Year','GP']); d=d[(d.ws<=0.10)&(~d.wet)]
d['ts']=d.TrackStatus.astype(str).str.split('.').str[0]
d['pure']=np.select([d.ts=='4',d.ts=='6',d.ts=='1'],['SC','VSC','G'],'mix')
d['code']=np.select([d.ts.str.contains('4'),d.ts.str.contains('6')],['SC','VSC'],'G')
nop=d[(d.PitIn==0)&(d.PitOut==0)&d.LapTime.notna()]
k=['Year','GP','LapNumber']
r1=nop[nop.pure!='mix'].groupby(k+['pure'])['LapTime'].agg(['median','size']).set_axis(['m1','n1'],axis=1)
r2=nop.groupby(k+['code'])['LapTime'].median().rename('m2')
r3=nop[nop.pure!='mix'].groupby(['Year','GP','pure'])['LapTime'].median().rename('m3')   # misma carrera, mismo estado puro
d=d.join(r1,on=k+['pure']).join(r2,on=k+['code']).join(r3,on=['Year','GP','pure'])
d['ref_src']=np.select([d.n1>=3,d.m2.notna(),d.m3.notna()],['vuelta_pura','vuelta','carrera'],'none')
d['med']=np.select([d.ref_src=='vuelta_pura',d.ref_src=='vuelta',d.ref_src=='carrera'],[d.m1,d.m2,d.m3],np.nan)
d['exc']=d.LapTime-d.med
d=d[(d.LapNumber>1)&(d.pure!='mix')&d.exc.notna()]
E=[]
for comp,flag in [('in','PitIn'),('out','PitOut')]:
    e=d[d[flag]==1].copy(); e['comp']=comp; E.append(e)
E=pd.concat(E); E['cls']=E.comp+'_'+E.pure; E=E[E.exc.abs()<60]
E.to_pickle(OUT+'clean_events.pkl')
t=E.groupby('cls')['exc'].agg(n='count',med='median',q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75))
t['GPs']=E.groupby('cls')['GP'].nunique(); t['races']=E.groupby('cls').apply(lambda s:s[['Year','GP']].drop_duplicates().shape[0])
print(t.round(2))
print('\nMediana por fuente de la referencia:'); print(E[E.pure!='G'].groupby(['cls','ref_src'])['exc'].agg(['count','median']).round(2))
print('\nn por GP del paper:'); print(E[E.GP.isin(PAPER)&(E.pure!='G')].groupby(['GP','cls']).size().unstack(fill_value=0))
