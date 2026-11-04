import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); sys.path.insert(0,".")
from models.regressions import all_laps, filtrar_outliers_iqr
laps=all_laps()
laps['GP']=laps['GP'].replace('Styrian Grand Prix','Austrian Grand Prix')
GPS=['Bahrain Grand Prix','Saudi Arabian Grand Prix','Miami Grand Prix','Hungarian Grand Prix',
     'Belgian Grand Prix','Dutch Grand Prix','Singapore Grand Prix','United States Grand Prix']
out=[]
for gp in GPS:
    pre=laps[(laps.GP==gp)&(laps.TrackStatus==1.0)]
    mod,post=filtrar_outliers_iqr(laps,gp)
    row={'GP':gp.replace(' Grand Prix','')}
    for k,c in (('in','PitIn'),('out','PitOut')):
        a=int((pre[c]==1).sum()); b=int((post[c]==1).sum())
        row[f'{k}_pre']=a; row[f'{k}_post']=b; row[f'{k}_drop%']=100*(a-b)/max(a,1)
    row['laps_pre']=len(pre); row['laps_post']=len(post)
    row['drop_all%']=100*(len(pre)-len(post))/len(pre)
    row['sigma']=float(np.sqrt(mod.mse_resid))
    # sd de los residuos en in/out laps SOBREVIVIENTES
    res=mod.resid
    for k,c in (('in','PitIn'),('out','PitOut')):
        m=(post[c]==1).to_numpy()
        row[f'{k}_sd_post']=float(np.std(res[m])) if m.sum()>3 else np.nan
    out.append(row)
d=pd.DataFrame(out)
pd.set_option('display.width',200)
print(d.round(2).to_string(index=False))
