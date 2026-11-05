import pandas as pd, numpy as np, glob, os, json, re
R='results/world2024_v10g_20261031/'
def load(d):
    return {os.path.basename(f)[:-4]:pd.read_csv(f) for f in sorted(glob.glob(R+d+'/*.csv'))}
F6,F7=load('fig6_2024_realdnf'),load('fig7_2024')
print('algos fig7',F7['Bahrain_ALO'].algo.value_counts().to_dict())
paper6={'Bahrain_ALO':[5.34,4.69,4.91,4.04,4.13,4.34],'Belgian_GAS':[10.74,10.71,11.79,10.77,10.80,11.18],'Dutch_STR':[7.67,7.98,7.83,6.79,6.70,7.72],'Hungarian_TSU':[8.47,8.37,8.28,7.73,7.84,8.02],'Miami_OCO':[12.98,13.85,14.50,12.76,13.06,12.99],'Saudi_Arabian_NOR':[4.61,4.45,4.71,4.41,4.26,4.51],'Singapore_HUL':[6.65,6.74,8.62,6.42,6.40,6.93],'United_States_MAG':[9.20,9.07,9.06,8.89,9.00,8.74]}
paper7={'Bahrain_ALO':[5.77,4.60,4.95,4.77,4.83,5.00],'Belgian_GAS':[12.19,12.19,12.37,12.30,12.89,12.31],'Dutch_STR':[11.06,9.59,9.65,9.61,9.83,10.73],'Hungarian_TSU':[8.55,8.46,8.45,8.45,9.54,9.35],'Miami_OCO':[10.58,11.01,12.62,11.07,12.57,12.27],'Saudi_Arabian_NOR':[7.20,5.34,5.66,5.25,5.25,5.26],'Singapore_HUL':[9.30,8.31,9.04,8.41,7.92,7.96],'United_States_MAG':[9.74,10.13,10.78,9.39,9.49,10.27]}
A=None
for name,F,P in (('F6',F6,paper6),('F7',F7,paper7)):
    algos=sorted(next(iter(F.values())).algo.unique()); print(name,algos)
    maxd=0;rows=[]
    for g,d in F.items():
        for a in d.algo.unique():
            s=d[d.algo==a].final_position; rows.append((g,a,s.mean(),s.std(),len(s),1.96*s.std()/np.sqrt(len(s))))
    T=pd.DataFrame(rows,columns='g a m sd n hw'.split())
    pv=T.pivot(index='g',columns='a',values='m'); 
    order=[c for c in ['DQN','A2C','TRPO','PPO','RecurrentPPO','RPPO','QMIP'] if c in pv]
    diff=(pv[order].round(2)-pd.DataFrame(P,index=order).T.reindex(pv.index)).abs()
    print(' max abs diff vs table',diff.max().max(),' n',T.n.unique()); 
    print(pv.mean().round(3).to_dict())
    print(' hw max diff? mean sd by algo',T.groupby('a').sd.mean().round(2).to_dict())
    if name=='F6': S6=T
    else: S7=T
m=S6.merge(S7,on=['g','a'],suffixes=('6','7')); m['w']=m.sd7/m.sd6
print('ratio sd7/sd6 by algo (mean over GP); share of GP-algo with sd7<sd6:',m.groupby('a').w.mean().round(3).to_dict(),(m.w<1).mean().round(2))
print(m.pivot(index='g',columns='a',values='w').round(2).to_string())
act=[c for c in F7['Bahrain_ALO'].algo.unique() if c.lower() in('actual','real','real_strategy','actual_strategy')]
print('act',act)
if act:
    a=act[0]
    pv7=S7.pivot(index='g',columns='a',values='m')
    print('replay mean',pv7[a].round(2).to_dict(),'avg',pv7[a].mean().round(3))
    for c in pv7: print(' diff vs replay',c,round(pv7[a].mean()-pv7[c].mean(),3))
    print('Belgian',pv7.loc['Belgian_GAS'].round(2).to_dict())
    print('sd actual box',S7[S7.a==a].set_index('g').sd.round(2).to_dict())
    for g,d in F7.items():
        s=d[d.algo==a].final_position; print(g,'q5,q95',s.quantile(.05),s.quantile(.95))
