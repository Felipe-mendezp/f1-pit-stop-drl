import os, numpy as np, pandas as pd, statsmodels.api as sm
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
exec(open(T+'ot_pair.py').read().split("GPS=['Bahrain'")[0])
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
D=np.array([-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20])
s=df[df.nosc_L1]
print('soporte gap<=1.0:', int((s.interval_front_real<=1.0).sum()),'de',len(s),'| adelantamientos en soporte:',int(s[s.interval_front_real<=1.0].ot_pair.sum()),'de',int(s.ot_pair.sum()))
rows=[]
for v,m in [('C',np.ones(len(s),bool)),('E',(s.interval_front_real<=1.0).values)]:
    for g in GPS:
        x=s[m & (s.GP==g+' Grand Prix').values]
        f=sm.Logit(x.ot_pair.values, sm.add_constant(x.delta_total.values)).fit(disp=False)
        a,b=f.params; rows.append(dict(v=v,GP=g,n=len(x),ot=int(x.ot_pair.sum()),const=a,slope=b,se=f.bse[1],**{f'{d:+d}':100/(1+np.exp(-(a+b*d))) for d in D}))
R=pd.DataFrame(rows); pd.set_option('display.width',250)
print(R.pivot(index='GP',columns='v',values=['n','ot','slope','se']).loc[GPS].round(3).to_string())
print('\n== E: probabilidad (%)'); print(R[R.v=='E'].set_index('GP')[[f'{d:+d}' for d in D]].round(1).to_string())
# en el soporte: tasa real vs modelo C y E por tramo
x=s[s.interval_front_real<=1.0].copy(); bins=[-30,-3,-1,-0.5,0,0.5,1,1.5,2,3,5]
for v in 'CE':
    P=R[R.v==v].set_index('GP'); g=x.GP.str.replace(' Grand Prix','')
    x['p'+v]=1/(1+np.exp(-(P.const.reindex(g).values+P.slope.reindex(g).values*x.delta_total.values)))
x['bin']=pd.cut(x.delta_total,bins)
print('\nen el soporte (8 GP juntos):'); print(x.groupby('bin',observed=False).agg(n=('ot_pair','size'),real=('ot_pair','mean'),modelo_C=('pC','mean'),modelo_E=('pE','mean')).round(3).T.to_string())
print('delta en soporte: percentiles', x.delta_total.quantile([.01,.05,.5,.95,.99]).round(2).to_dict())
R.to_csv(T+'ot_support.csv',index=False)
