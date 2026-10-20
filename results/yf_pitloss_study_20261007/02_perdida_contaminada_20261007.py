from yf_common_20261007 import *
d=load()
d=d[~d.wet]
d=d[~d.GP.isin([])]
# mediana de vueltas sin pit por carrera-vuelta
nopit=d[(d.PitIn==0)&(d.PitOut==0)]
med=nopit.groupby(['Year','GP','LapNumber'])['LapTime'].median().rename('med')
d=d.join(med,on=['Year','GP','LapNumber'])
d['exc']=d.LapTime-d.med
d=d.sort_values(['Year','GP','Driver','LapNumber'])
g=d.groupby(['Year','GP','Driver'])
d['next_exc']=g['exc'].shift(-1); d['next_out']=g['PitOut'].shift(-1); d['next_st']=g['lap_st'].shift(-1)
st=d[(d.PitIn==1)&(d.next_out==1)&(d.LapNumber>1)].copy()
st['loss']=st.exc+st.next_exc
st['cls']=np.where(st.lap_st=='G','G',st.lap_st+np.where(st.ev_idx==1,'1','2+'))
st=st[st.lap_st.isin(['G','SC','VSC'])&st.loss.notna()]
st.to_pickle(OUT+'stops.pkl')
q=lambda s: pd.Series({'n':len(s),'med':s.median(),'iqr':s.quantile(.75)-s.quantile(.25),'mean':s.mean(),'sd':s.std()})
print(st.groupby('cls')['loss'].apply(q).unstack().round(2))
print('\nin-lap / out-lap excess por clase (mediana):')
print(st.groupby('cls')[['exc','next_exc']].median().round(2))
# por GP del paper
gl=st[st.cls=='G'].groupby('GP')['loss'].median().rename('G')
t=st.groupby(['GP','cls'])['loss'].agg(['median','count']).unstack()
tab=pd.concat([t['median'].round(1),t['count']],axis=1,keys=['med','n'])
print('\nPor GP (paper):'); print(tab.loc[PAPER].to_string())
# ratio YF/G por GP con >=3 obs
r=t['median'].div(t['median']['G'],axis=0)
ok=t['count']>=3
print('\nRatio pérdida YF / verde (GPs con >=3 paradas), resumen:')
for c in ['SC1','SC2+','VSC1','VSC2+']:
    v=r[c][ok[c]]
    print(c, 'nGP',len(v),'med',round(v.median(),2),'IQR',round(v.quantile(.25),2),round(v.quantile(.75),2))
