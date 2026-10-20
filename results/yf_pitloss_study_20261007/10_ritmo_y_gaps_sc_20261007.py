"""2026-10-07 — (a) ritmo de vuelta SC/VSC sin pit (razón sobre la vuelta verde de referencia), 2021-23,
agrupado con contracción por GP; (b) gap al auto de adelante bajo SC según vuelta del evento (2021-24)."""
from yf_common_20261007 import *
d=load(); ws=d.groupby(['Year','GP'])['wet'].mean(); d=d.join(ws.rename('ws'),on=['Year','GP']); d=d[(d.ws<=0.10)&(~d.wet)]
d['ts']=d.TrackStatus.astype(str).str.split('.').str[0]
d['pure']=np.select([d.ts=='4',d.ts=='6'],['SC','VSC'],'x')
nop=d[(d.PitIn==0)&(d.PitOut==0)&(d.LapNumber>1)&d.LapTime.notna()]
# (a) ritmo: mediana por carrera-vuelta de autos sin pit con vuelta pura
p=nop[nop.pure!='x'].groupby(['Year','GP','LapNumber','pure','ev_idx']).agg(laptime=('LapTime','median'),n=('LapTime','size'),Lref=('Lref','first')).reset_index()
p=p[p.n>=3]; p['rho']=p.laptime/p.Lref
p['k']=np.where(p.pure=='VSC','VSC',np.where(p.ev_idx<=1,'SC_1',np.where(p.ev_idx==2,'SC_2',np.where(p.ev_idx==3,'SC_3','SC_4+'))))
print('(a) Ritmo (vuelta sin pit / vuelta verde), vueltas 100% bajo el estado:')
for yrs,lab in [((2021,2022,2023),'2021-23'),((2021,2022,2023,2024),'2021-24')]:
    q=p[p.Year.isin(yrs)]
    print(lab); print(q.groupby('k').agg(vueltas=('rho','size'),carreras=('Year',lambda s: q.loc[s.index,['Year','GP']].drop_duplicates().shape[0]),med=('rho','median'),q25=('rho',lambda s:s.quantile(.25)),q75=('rho',lambda s:s.quantile(.75))).round(3))
q=p[p.Year<=2023]
# contracción por GP (empirical Bayes simple) para SC vueltas 2-4+ agrupadas y VSC
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
Lref=d[d.Year<=2023].groupby('GP')['Lref'].median()
rows=[]
for st,mask in [('SC',q.k.isin(['SC_2','SC_3','SC_4+'])),('VSC',q.k=='VSC')]:
    s=q[mask]; mu=s.rho.median(); gm=s.groupby('GP').rho.agg(['mean','size'])
    sw=s.groupby('GP').rho.var().median(); tau2=max(gm['mean'].var()-sw/gm['size'].mean(),1e-4)
    for gp in PAPER:
        n=gm['size'].get(gp,0); raw=gm['mean'].get(gp,np.nan); w=n/(n+sw/tau2) if n else 0
        rows.append(dict(GP=gp.replace(' Grand Prix',''),est=st,n=n,raw=raw,shrunk=w*(raw if n else 0)+(1-w)*mu,
                         sim=x.loc[gp,'yf_'+st]/x.loc[gp,'reg_intercept']))
R=pd.DataFrame(rows).pivot(index='GP',columns='est',values=['n','raw','shrunk','sim']).round(2)
print('\nPor GP del paper (2021-23), razón sobre vuelta verde; sim = yf_SC|VSC / reg_intercept 2024:'); print(R.to_string())
# (b) gaps bajo SC
g=nop[(nop.pure=='SC')&(nop.Interval_front<50)&(nop.Position>1)]
print('\n(b) Gap al de adelante (Interval_front, s) bajo SC, autos sin pit, vuelta 100% SC, P>1:')
g['k']=g.ev_idx.clip(upper=6)
print(g.groupby('k')['Interval_front'].describe(percentiles=[.1,.25,.5,.75,.9])[['count','10%','25%','50%','75%','90%']].round(2))
gg=g[g.ev_idx>=3]
print('\nVuelta >=3 del SC, mediana por posición (P2..P20):'); print(gg.groupby('Position')['Interval_front'].median().round(2).to_dict())
print('Vuelta >=3: mediana por carrera (distribución):', gg.groupby(['Year','GP'])['Interval_front'].median().describe().round(2).to_dict())
print('\nComparación verde (vueltas verdes, sin pit, P>1): mediana',round(nop[(nop.ts=='1')&(nop.Position>1)&(nop.Interval_front<50)].Interval_front.median(),2))
