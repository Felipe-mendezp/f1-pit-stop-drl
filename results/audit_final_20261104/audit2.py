import pandas as pd, numpy as np, glob, os, json, re
D='results/decision_support_v10gw24belief_20261103/PPO/'; pre='saudi_arabian_nor_real_lap'
d=pd.read_csv(D+pre+'7.csv'); m=json.load(open(D+pre+'7_meta.json'))
print(m['recommended_label'],m['agent_state_at_lap'],m['M'],m['available_compounds'],m['prefix_yf_laps'], 'hist first',m['history'][:2])
g=d.groupby('action_label'); print(g.agg(pos=('final_position','mean'),pts=('points','mean'),top10=('final_position',lambda s:(s<=10).mean()),stops=('n_stops','mean')).round(3))
piv=d.pivot(index='rollout',columns='action_label',values='points'); rec=m['recommended_label']
for c in piv:
    if c!=rec:
        df=piv[c]-piv[rec]; print(c,'dif',round(df.mean(),3),'2se',round(2*df.std()/np.sqrt(len(df)),3))
np_=d[d.action==0]; pl=np_.pit_laps.fillna('').astype(str).str.split(';').apply(lambda x:[int(float(i)) for i in x if i])
first=pl.apply(lambda x:min([i for i in x if i>7],default=None))
print('stay out: stop laps desc',first.describe().round(1).to_dict(),'share first stop >25:',(first>25).mean(),'share>=26 of those with stop', (first.dropna()>25).mean(),'none',first.isna().mean())
print('stay: n distinct lt 26:',(first<=25).sum())
# evolution
rows=[]
for f in sorted(glob.glob(D+pre+'*[0-9].csv'),key=lambda s:int(re.findall(r'lap(\d+)',s)[0])):
    n=int(re.findall(r'lap(\d+)',f)[0]); x=pd.read_csv(f); mm=json.load(open(f.replace('.csv','_meta.json')))
    p=x.groupby('action_label').points.mean().round(2).to_dict(); p['n']=n; p['rec']=mm['recommended_label']; rows.append(p)
E=pd.DataFrame(rows).set_index('n'); print(E.head(12).to_string()); 
c=[k for k in E.columns if k!='rec']; 
noact=[k for k in c if 'No' in k][0]
late=E.loc[10:]; print('laps>=10 where some stop action > no pit by>0:',[(n,round(late.loc[n,[k for k in c if k!=noact]].max()-late.loc[n,noact],2)) for n in late.index if late.loc[n,[k for k in c if k!=noact]].max()>late.loc[n,noact]][:15])
print('rec by lap non-No pit:',E[E.rec!=noact].rec.to_dict())
# Fig11
L=pd.read_csv('results/decision_support_v10gsmrealhist_20261102/compare_laps_20261103.csv')
print(L.groupby(['scen','algo']).size())
for s,x in L.groupby('scen'):
    print(s,'rec=real',round((x.real==x.rec).mean()*100,1),'n',len(x))
    y=x[x.real!=x.rec].assign(g=lambda t:t.pts_rec-t.pts_real); y['gg']=-y.dpts_real_vs_rec
    print(y.sort_values('gg',ascending=False)[['gp','lap','real','rec','gg','dpts_2se','signif','comp']].head(8).to_string())
    print(' sum net by gp',y.groupby('gp').gg.sum().round(2).to_dict())
x=L[(L.scen=='(ii)')&(L.gp=='bahrain_alo')&(L.lap.between(33,43))][['lap','real','rec','dpts_real_vs_rec','dpts_2se','signif']]; print(x.to_string())
print(L[(L.scen=='(ii)')&(L.lap==1)][['gp','comp','tyrelife','pos']].to_string())
