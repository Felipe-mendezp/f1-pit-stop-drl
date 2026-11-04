import pandas as pd, numpy as np
fs=[f"data/session_{y}_V2.csv" for y in (2021,2022,2023,2024)]
df=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
df['TS']=df.TrackStatus.astype(str)
df=df.dropna(subset=['LapTime'])
df=df.sort_values(['Year','GP','Driver','LapNumber'])

rows=[]
for (y,gp,drv),g in df.groupby(['Year','GP','Driver'],sort=False):
    g=g.reset_index(drop=True)
    clean=(g.PitIn==0)&(g.PitOut==0)&(g.TS=='1')
    lt=g.LapTime.to_numpy(float)
    for i in range(len(g)):
        for kind,flag in (('in',g.PitIn.iat[i]),('out',g.PitOut.iat[i])):
            if flag!=1 or g.TS.iat[i]!='1': continue
            # referencia: vueltas limpias cercanas del mismo piloto (ventana +-5, excluyendo pit laps)
            lo,hi=max(0,i-5),min(len(g),i+6)
            idx=[j for j in range(lo,hi) if j!=i and clean.iat[j]]
            if len(idx)<3: continue
            ref=np.median(lt[idx])
            rows.append((y,gp,drv,kind,lt[i]-ref,lt[i],ref))
r=pd.DataFrame(rows,columns=['Year','GP','Driver','kind','loss','lap','ref'])
print("n por tipo:",r.kind.value_counts().to_dict())
for k in ('in','out'):
    s=r[r.kind==k].loss
    q=s.quantile([.01,.05,.25,.5,.75,.9,.95,.99]).round(2).to_dict()
    print(f"\n[{k}-lap] n={len(s)} media={s.mean():.2f} sd={s.std():.2f}")
    print("  cuantiles:",{f"p{int(a*100)}":b for a,b in q.items()})
    print(f"  max={s.max():.1f}  >+10s sobre mediana: {(s>s.median()+10).mean()*100:.1f}%  >+5s: {(s>s.median()+5).mean()*100:.1f}%")
r.to_csv("results/pitstop_var_20261031/pit_loss.csv",index=False)
