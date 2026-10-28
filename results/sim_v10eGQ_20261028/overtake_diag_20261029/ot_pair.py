import os, numpy as np, pandas as pd, statsmodels.api as sm
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
exec(open(T+'ot_variants.py').read().split('V = {')[0].replace("print(len(bad_))","pass").replace("print(bad_[","(bad_[").replace("print('etiqueta","('etiqueta"))
df['ot_pair']=((df.next_position.notna()) & (pd.Series(r_np,index=df.index).notna()) & (df.next_position < r_np) & (r_last != df.LapNumber)).astype(int)
s=df[df.nosc_L1]
print('muestra C:',len(s),'| etiqueta actual (sin estado):',int(s.ot_any.sum()),'| por par:',int(s.ot_pair.sum()),
      '| por par y no actual:',int(((s.ot_pair==1)&(s.ot_any==0)).sum()),'| actual y no por par:',int(((s.ot_pair==0)&(s.ot_any==1)).sum()))
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
D=np.array([-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20]); bins=[-30,-5,-3,-2,-1,-0.5,0,0.5,1,2,3,5]
rows=[]; emp={}
for g in GPS:
    x=s[s.GP==g+' Grand Prix'].copy()
    f=sm.Logit(x.ot_pair.values, sm.add_constant(x.delta_total.values)).fit(disp=False)
    a,b=f.params; rows.append(dict(GP=g,n=len(x),ot=int(x.ot_pair.sum()),const=a,slope=b,**{f'{d:+d}':100/(1+np.exp(-(a+b*d))) for d in D}))
    x['p']=f.predict(sm.add_constant(x.delta_total.values)); x['bin']=pd.cut(x.delta_total,bins)
    t=x.groupby('bin',observed=False).agg(n=('ot_pair','size'),real=('ot_pair','mean'),modelo=('p','mean'))
    emp[g]=t
R=pd.DataFrame(rows).set_index('GP'); pd.set_option('display.width',250)
print(R[['n','ot','const','slope']].round(3).to_string()); print(R[[f'{d:+d}' for d in D]].round(1).to_string())
for g in ['Belgian','Bahrain','Saudi Arabian']:
    t=emp[g]; print('\n==',g); print(t.assign(real=100*t.real,modelo=100*t.modelo).round(1).T.to_string())
R.to_csv(T+'ot_pair_table.csv')
